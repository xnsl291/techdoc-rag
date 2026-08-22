"""Step 1 - Local LLM 실측 벤치마크.

노션 `07. 이어서 작업하기` Step 1이 요구하는 표를 만든다.

    모델 / 양자화 / thinking / TTFT / tok-s / RAM peak / 10회 후 tok-s

이 파일은 일회성 실측 도구다. 애플리케이션 코드가 아니므로 scripts/ 아래 둔다.
여기서 나온 값만 설계 근거로 쓸 수 있고, 그 전까지의 수치는 전부 추정이다.

사용법:
    python scripts/benchmark_llm.py --models qwen3.5:4b-q4_K_M --repeat 10
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434

# 실제 기술문서 질의와 성격이 비슷해야 의미가 있으므로 인버터 도메인 질문을 쓴다.
# 근거 문서를 붙이지 않은 순수 생성 속도 측정이므로 Retrieval 품질과는 무관하다.
BENCHMARK_PROMPT = (
    "산업용 인버터에서 과전류 트립이 반복해서 발생할 때 "
    "점검해야 할 항목을 순서대로 설명해 주세요."
)


@dataclass
class RunResult:
    """단일 생성 요청 1회의 측정값."""

    model: str
    thinking_enabled: bool
    time_to_first_token_seconds: float
    generated_token_count: int
    tokens_per_second: float
    prompt_token_count: int
    prompt_eval_seconds: float
    total_seconds: float


class MemorySampler:
    """생성 중 ollama 관련 프로세스의 RSS 합계를 주기적으로 표본화한다.

    Windows에서 Ollama는 서버 프로세스와 모델 러너 프로세스가 분리되므로
    이름이 ollama로 시작하는 프로세스를 모두 합산한다.
    """

    def __init__(self, sample_interval_seconds: float = 0.25) -> None:
        self._sample_interval_seconds = sample_interval_seconds
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int:
        total_bytes = 0
        for process in psutil.process_iter(["name"]):
            process_name = (process.info["name"] or "").lower()
            if process_name.startswith("ollama"):
                try:
                    total_bytes += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return total_bytes

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._peak_bytes = max(self._peak_bytes, self._sample_once())
            self._stop_event.wait(self._sample_interval_seconds)

    def __enter__(self) -> MemorySampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exception_info: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_megabytes(self) -> float:
        return self._peak_bytes / (1024 * 1024)


class OllamaHttpClient:
    """Ollama 호출을 하나의 TCP 연결로 재사용한다.

    요청마다 새 소켓을 열면 Windows에서 TIME_WAIT 소켓이 쌓이고,
    동적 포트 범위(기본 49152~65535, 16,384개)가 고갈되면 connect 자체가
    WinError 10048로 실패한다. 2026-08-22 이 장비에서 실제로 발생했다.
    애플리케이션의 LlmClient 어댑터도 같은 이유로 연결을 재사용해야 한다.
    """

    # Windows 동적 포트 범위(49152~65535)가 TIME_WAIT 소켓으로 고갈되면
    # 커널이 출발지 포트를 할당하지 못해 connect가 WinError 10048로 실패한다.
    # 평소에는 커널에 맡기고, 그 상황에서만 동적 범위 밖의 포트로 우회한다.
    FALLBACK_SOURCE_PORTS = range(30000, 30050)

    def __init__(self, timeout_seconds: int = 900) -> None:
        self._timeout_seconds = timeout_seconds
        self._connection = self._connect()

    def _open(self, source_port: int | None) -> http.client.HTTPConnection:
        source_address = (OLLAMA_HOST, source_port) if source_port else None
        connection = http.client.HTTPConnection(
            OLLAMA_HOST, OLLAMA_PORT, timeout=self._timeout_seconds, source_address=source_address
        )
        connection.connect()
        return connection

    def _connect(self) -> http.client.HTTPConnection:
        try:
            return self._open(source_port=None)
        except OSError as error:
            print(f"  기본 연결 실패({error}). 출발지 포트를 지정해 재시도", flush=True)

        for source_port in self.FALLBACK_SOURCE_PORTS:
            try:
                connection = self._open(source_port)
                print(f"  출발지 포트 {source_port} 사용", flush=True)
                return connection
            except OSError:
                continue
        raise RuntimeError("Ollama 연결 실패. 동적 포트 고갈 여부를 확인할 것")

    def get_json(self, path: str) -> dict:
        self._connection.request("GET", path)
        return json.loads(self._connection.getresponse().read())

    def post_stream(self, path: str, payload: dict) -> http.client.HTTPResponse:
        self._connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return self._connection.getresponse()

    def close(self) -> None:
        self._connection.close()


def list_installed_models(client: OllamaHttpClient) -> list[str]:
    return [model["name"] for model in client.get_json("/api/tags").get("models", [])]


def run_single_generation(
    client: OllamaHttpClient,
    model: str,
    thinking_enabled: bool,
    prompt: str,
    context_length: int,
    max_generated_tokens: int,
) -> RunResult:
    """스트리밍으로 1회 생성하고 첫 토큰 지연과 생성 속도를 측정한다.

    tokens/sec은 Ollama가 돌려주는 eval_count / eval_duration을 쓴다.
    벽시계로 나누면 네트워크와 파싱 오버헤드가 섞여 모델 자체 속도가 흐려진다.

    max_generated_tokens로 길이를 고정하는 이유는 두 가지다.
    조건별 생성량이 달라지면 tok/s는 비교 가능해도 총 소요시간이 비교 불가가 되고,
    thinking ON에서 생성이 길어지면 측정 자체가 몇 시간 단위로 늘어난다.
    실제 답변 길이에서의 체감은 Step 6에서 따로 확인한다.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": thinking_enabled,
        "options": {
            "temperature": 0,
            "num_ctx": context_length,
            "num_predict": max_generated_tokens,
        },
    }
    started_at = time.perf_counter()
    time_to_first_token_seconds: float | None = None
    final_chunk: dict = {}

    response = client.post_stream("/api/generate", payload)
    if response.status != 200:
        raise RuntimeError(f"Ollama HTTP {response.status}: {response.read().decode()[:300]}")

    for raw_line in response:
        if not raw_line.strip():
            continue
        chunk = json.loads(raw_line)
        # thinking 모델은 사고 과정을 response가 아니라 thinking 필드로 흘린다.
        # 사용자 체감 기준으로는 둘 중 무엇이든 첫 출력이 나온 시점이 TTFT다.
        has_output = bool(chunk.get("response")) or bool(chunk.get("thinking"))
        if has_output and time_to_first_token_seconds is None:
            time_to_first_token_seconds = time.perf_counter() - started_at
        if chunk.get("done"):
            final_chunk = chunk

    total_seconds = time.perf_counter() - started_at
    generated_token_count = final_chunk.get("eval_count", 0)
    eval_duration_nanoseconds = final_chunk.get("eval_duration", 0)
    tokens_per_second = (
        generated_token_count / (eval_duration_nanoseconds / 1_000_000_000)
        if eval_duration_nanoseconds
        else 0.0
    )

    return RunResult(
        model=model,
        thinking_enabled=thinking_enabled,
        time_to_first_token_seconds=time_to_first_token_seconds or total_seconds,
        generated_token_count=generated_token_count,
        tokens_per_second=tokens_per_second,
        prompt_token_count=final_chunk.get("prompt_eval_count", 0),
        prompt_eval_seconds=final_chunk.get("prompt_eval_duration", 0) / 1_000_000_000,
        total_seconds=total_seconds,
    )


def benchmark_condition(
    client: OllamaHttpClient,
    model: str,
    thinking_enabled: bool,
    repeat_count: int,
    context_length: int,
    max_generated_tokens: int,
) -> dict:
    """모델 x thinking 한 조건을 repeat_count회 반복 측정한다.

    첫 회는 모델 적재 시간이 섞이므로 워밍업으로 따로 두고 통계에서 제외한다.
    마지막 회를 별도로 남기는 이유는 노트북의 thermal throttling 확인이다.
    """
    label = f"{model} / thinking={'ON' if thinking_enabled else 'OFF'}"
    print(f"\n[{label}] 워밍업 중...", flush=True)

    with MemorySampler() as sampler:
        warmup_result = run_single_generation(
            client,
            model,
            thinking_enabled,
            BENCHMARK_PROMPT,
            context_length,
            max_generated_tokens,
        )
        runs: list[RunResult] = []
        for index in range(repeat_count):
            result = run_single_generation(
                client,
                model,
                thinking_enabled,
                BENCHMARK_PROMPT,
                context_length,
                max_generated_tokens,
            )
            runs.append(result)
            print(
                f"  {index + 1:2d}/{repeat_count}  "
                f"TTFT {result.time_to_first_token_seconds:5.2f}s  "
                f"{result.tokens_per_second:5.1f} tok/s  "
                f"{result.generated_token_count:4d} tokens",
                flush=True,
            )

    return {
        "model": model,
        "thinking_enabled": thinking_enabled,
        "repeat_count": repeat_count,
        "context_length": context_length,
        "max_generated_tokens": max_generated_tokens,
        "warmup_time_to_first_token_seconds": round(
            warmup_result.time_to_first_token_seconds, 3
        ),
        "median_time_to_first_token_seconds": round(
            statistics.median(run.time_to_first_token_seconds for run in runs), 3
        ),
        # 중앙값만 보면 체감을 놓친다. 상용 서비스도 P95가 P50의 2배 안팎이므로
        # 최악값을 함께 남긴다. 표본 10회에서 P95는 사실상 최댓값이라 그대로 최댓값을 쓴다.
        "worst_time_to_first_token_seconds": round(
            max(run.time_to_first_token_seconds for run in runs), 3
        ),
        "median_tokens_per_second": round(
            statistics.median(run.tokens_per_second for run in runs), 2
        ),
        "worst_tokens_per_second": round(
            min(run.tokens_per_second for run in runs), 2
        ),
        "first_run_tokens_per_second": round(runs[0].tokens_per_second, 2),
        "last_run_tokens_per_second": round(runs[-1].tokens_per_second, 2),
        "median_generated_token_count": int(
            statistics.median(run.generated_token_count for run in runs)
        ),
        "median_total_seconds": round(
            statistics.median(run.total_seconds for run in runs), 2
        ),
        "peak_memory_megabytes": round(sampler.peak_megabytes, 1),
        "runs": [asdict(run) for run in runs],
    }


def format_markdown_table(summaries: list[dict]) -> str:
    header = (
        "| 모델 | thinking | TTFT 중앙값 | TTFT 최악 | tok/s 중앙값 | tok/s 최악 "
        "| 생성토큰 | 총 소요 | RAM peak | 1회차 tok/s | 마지막 tok/s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        "| {model} | {thinking} | {ttft:.2f}s | {ttft_worst:.2f}s | {tokens_per_second:.1f} "
        "| {tokens_per_second_worst:.1f} | {token_count} | {total:.1f}s | {memory:,.0f} MB "
        "| {first:.1f} | {last:.1f} |".format(
            model=summary["model"],
            thinking="ON" if summary["thinking_enabled"] else "OFF",
            ttft=summary["median_time_to_first_token_seconds"],
            ttft_worst=summary["worst_time_to_first_token_seconds"],
            tokens_per_second=summary["median_tokens_per_second"],
            tokens_per_second_worst=summary["worst_tokens_per_second"],
            token_count=summary["median_generated_token_count"],
            total=summary["median_total_seconds"],
            memory=summary["peak_memory_megabytes"],
            first=summary["first_run_tokens_per_second"],
            last=summary["last_run_tokens_per_second"],
        )
        for summary in summaries
    ]
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM 생성 속도 실측")
    parser.add_argument("--models", nargs="+", required=True, help="측정할 Ollama 모델 태그")
    parser.add_argument("--repeat", type=int, default=10, help="조건당 반복 횟수")
    parser.add_argument("--context-length", type=int, default=8192, help="num_ctx 값")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="조건별 생성 길이 고정값 (num_predict)"
    )
    parser.add_argument(
        "--output-directory", default="scripts/output", help="결과 JSON 저장 위치"
    )
    arguments = parser.parse_args()

    client = OllamaHttpClient()
    installed_models = list_installed_models(client)
    missing_models = [model for model in arguments.models if model not in installed_models]
    if missing_models:
        print("아래 모델이 설치되어 있지 않습니다. 먼저 pull 하십시오:")
        for model in missing_models:
            print(f"  ollama pull {model}")
        raise SystemExit(1)

    summaries = []
    try:
        for model in arguments.models:
            for thinking_enabled in (False, True):
                summaries.append(
                    benchmark_condition(
                        client,
                        model,
                        thinking_enabled,
                        arguments.repeat,
                        arguments.context_length,
                        arguments.max_tokens,
                    )
                )
    finally:
        client.close()

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "benchmark_llm.json"
    result_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + format_markdown_table(summaries))
    print(f"\n원본 결과: {result_path}")


if __name__ == "__main__":
    main()
