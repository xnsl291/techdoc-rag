"""Ollama LLM 클라이언트 (DP-42, DP-51).

ports.LlmClient 계약의 요구를 그대로 구현한다. 스트리밍이 기본이고,
커넥션 재사용·동시 생성 제한·타임아웃·취소 처리가 전부 계약에 정의되어 있다.

thinking 토큰은 내보내지 않는다. Ollama는 thinking을 response가 아닌 별도
필드로 흘리는데, 답변 이터레이터에 사고 과정이 섞이면 Citation 대상이
오염된다. 다만 시간에는 포함된다 — 사용자가 실제로 기다리는 시간이다.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit

from techdoc_rag.domain.errors import GenerationError

_GENERATE_PATH = "/api/generate"


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """한 번의 생성에서 측정한 지연(NFR-005).

    wait는 세마포어 대기, first_token은 요청 전송부터 첫 출력(thinking 포함)까지,
    total은 요청 전송부터 스트림 종료까지다. 호출자가 재는 "첫 토큰까지"에는
    wait가 섞이므로 구분해서 남긴다.
    """

    wait_seconds: float
    first_token_seconds: float | None
    total_seconds: float
    completed: bool  # done을 받고 끝났는가. False면 취소·오류로 중단된 것


class OllamaLlmClient:
    """Ollama /api/generate 스트리밍 클라이언트.

    last_metrics는 마지막 generate의 측정값이다. 동시 생성 1(기본)에서는
    안전하지만, 값을 올리면 마지막으로 끝난 생성의 값만 남는다.
    """

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        temperature: float,
        runtime_context_tokens: int,
        thinking_enabled: bool,
        max_concurrent_generations: int,
        queue_timeout_seconds: float,
        generation_timeout_seconds: float,
    ) -> None:
        parts = urlsplit(endpoint)
        if parts.hostname == "localhost":
            raise GenerationError(
                "endpoint에 localhost를 쓰지 말 것. IPv6 폴백으로 지연이 붙는다. "
                "127.0.0.1을 쓸 것"
            )
        if parts.hostname is None or parts.port is None:
            raise GenerationError(f"endpoint 형식이 잘못됨: {endpoint} (예: http://127.0.0.1:11434)")
        if max_concurrent_generations != 1:
            # 이 구현은 커넥션 하나를 재사용하므로 동시 생성 2 이상이면
            # 스레드들이 같은 커넥션을 공유해 요청과 응답이 섞인다(리뷰 지적).
            # DP-51로 값을 올리는 날에는 커넥션 풀부터 만들어야 한다.
            # 조용히 깨지는 것보다 여기서 막는 쪽을 택한다.
            raise GenerationError(
                "이 구현은 max_concurrent_generations=1만 지원함. "
                "값을 올리려면 커넥션 분리(풀)가 먼저 필요함"
            )

        self._model_name = model_name
        self._host = parts.hostname
        self._port = parts.port
        self._temperature = temperature
        self._runtime_context_tokens = runtime_context_tokens
        self._thinking_enabled = thinking_enabled
        self._queue_timeout_seconds = queue_timeout_seconds
        self._generation_timeout_seconds = generation_timeout_seconds
        self._semaphore = threading.Semaphore(max_concurrent_generations)
        self._connection: http.client.HTTPConnection | None = None
        self._last_metrics: GenerationMetrics | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def last_metrics(self) -> GenerationMetrics | None:
        return self._last_metrics

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]:
        """토큰 조각을 순서대로 내놓는 제너레이터.

        다 쓰지 않을 이터레이터는 명시적으로 close()할 것 — GC에 맡기면
        세마포어 해제와 서버 쪽 생성 중단 시점이 불확정이다.

        제너레이터라 대기·접속은 첫 next()에서 일어난다. 대기 초과의
        GenerationError는 generate() 호출부가 아니라 첫 소비 지점에서 터진다.
        """
        wait_started = time.perf_counter()
        if not self._semaphore.acquire(timeout=self._queue_timeout_seconds):
            raise GenerationError(
                f"생성 대기 초과: {self._queue_timeout_seconds}초. "
                f"동시 생성 한도에 막혀 있음 (DP-51)"
            )
        wait_seconds = time.perf_counter() - wait_started

        request_started = time.perf_counter()
        first_token_at: float | None = None
        completed = False
        try:
            response = self._send(prompt, max_tokens)
            while True:
                self._check_deadline(request_started)
                line = self._read_line(response)
                if not line:
                    break
                payload = self._parse_line(line)
                if "error" in payload:
                    raise GenerationError(f"생성 실패: {payload['error']}")
                # thinking이든 답변이든 첫 출력이 나온 시점이 체감 첫 반응이다.
                if first_token_at is None and (
                    payload.get("response") or payload.get("thinking")
                ):
                    first_token_at = time.perf_counter()
                token = payload.get("response", "")
                if token:
                    yield token
                if payload.get("done"):
                    # 종료 마커(빈 chunk)까지 읽어 응답을 EOF로 만든다.
                    # 남기면 http.client가 이 커넥션의 다음 요청을 거부한다.
                    self._drain(response)
                    completed = True
                    break
        finally:
            if not completed:
                # 읽다 만 바이트가 남은 커넥션을 재사용하면 다음 응답의
                # 상태줄을 잘못 읽는다. 취소·오류·타임아웃이면 버린다.
                self.close()
            total = time.perf_counter() - request_started
            self._last_metrics = GenerationMetrics(
                wait_seconds=wait_seconds,
                first_token_seconds=(
                    first_token_at - request_started if first_token_at is not None else None
                ),
                total_seconds=total,
                completed=completed,
            )
            self._semaphore.release()

    def _send(self, prompt: str, max_tokens: int) -> http.client.HTTPResponse:
        payload = json.dumps(
            {
                "model": self._model_name,
                "prompt": prompt,
                "stream": True,
                "think": self._thinking_enabled,
                "options": {
                    "temperature": self._temperature,
                    "num_ctx": self._runtime_context_tokens,
                    "num_predict": max_tokens,
                },
            }
        ).encode("utf-8")
        try:
            connection = self._ensure_connection()
            connection.request(
                "POST",
                _GENERATE_PATH,
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
        except (http.client.HTTPException, ConnectionError, TimeoutError, OSError) as exc:
            self.close()
            raise GenerationError(f"LLM 서버 접속 실패: {self._host}:{self._port} ({exc})") from exc
        if response.status != 200:
            body = response.read()[:200]
            self.close()
            raise GenerationError(f"생성 요청 실패: HTTP {response.status} {body!r}")
        return response

    def _drain(self, response: http.client.HTTPResponse) -> None:
        try:
            response.read()
        except (http.client.HTTPException, ConnectionError, TimeoutError, OSError):
            # 소진에 실패한 커넥션은 재사용할 수 없다. 생성 자체는 이미 끝났으므로
            # 오류로 올리지 않고 커넥션만 버린다.
            self.close()

    def _read_line(self, response: http.client.HTTPResponse) -> bytes:
        try:
            return response.readline()
        except (http.client.HTTPException, ConnectionError, TimeoutError, OSError) as exc:
            raise GenerationError(f"스트림 읽기 실패: {exc}") from exc

    def _check_deadline(self, request_started: float) -> None:
        if time.perf_counter() - request_started > self._generation_timeout_seconds:
            raise GenerationError(
                f"생성 시간 초과: {self._generation_timeout_seconds}초. "
                f"모델이 멈췄거나 답변이 예산보다 김"
            )

    @staticmethod
    def _parse_line(line: bytes) -> dict:
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"스트림 응답이 JSON이 아님: {line[:100]!r}") from exc

    def _ensure_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            # 소켓 타임아웃은 "한 줄 읽기가 멈추는" 경우를, _check_deadline은
            # "느리게 계속 나오지만 총 시간이 넘는" 경우를 잡는다.
            self._connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self._generation_timeout_seconds
            )
        return self._connection
