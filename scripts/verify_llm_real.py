"""실제 Ollama로 LLM 클라이언트를 검증한다 (#13의 실물 항목).

단위 테스트가 못 보는 것을 본다.

1. 실제 스트리밍에서 토큰이 조각으로 도착하고 지연이 분리 기록되는가
2. 스트림을 중간에 닫은 뒤 다음 호출이 정상 동작하는가 (커넥션 폐기·재수립)
3. 없는 모델이 GenerationError로 오는가
4. thinking ON/OFF의 체감 차이 (이 기기 값. 기기 종속)

사용:
    python scripts/verify_llm_real.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techdoc_rag.adapters.ollama_llm_client import OllamaLlmClient  # noqa: E402
from techdoc_rag.domain.errors import GenerationError  # noqa: E402

PROMPT = "인버터의 과전류 보호 기능을 세 문장으로 설명하라."


def _client(model_name: str, thinking: bool) -> OllamaLlmClient:
    return OllamaLlmClient(
        model_name=model_name,
        endpoint="http://127.0.0.1:11434",
        temperature=0.0,
        runtime_context_tokens=8192,
        thinking_enabled=thinking,
        max_concurrent_generations=1,
        queue_timeout_seconds=120,
        generation_timeout_seconds=300,
    )


def main() -> int:
    print(f"기기: {platform.node()}")

    # 1) 스트리밍과 지연 분리
    client = _client("qwen3.5:4b-q4_K_M", thinking=False)
    pieces = list(client.generate(PROMPT, max_tokens=256))  # 워밍업 겸
    pieces = list(client.generate(PROMPT, max_tokens=256))
    metrics = client.last_metrics
    assert metrics is not None
    print(
        f"\n[1] 스트리밍: 조각 {len(pieces)}개 / wait {metrics.wait_seconds * 1000:.0f}ms"
        f" / 첫 출력 {metrics.first_token_seconds:.2f}s"
        f" / 총 {metrics.total_seconds:.2f}s / 완주 {metrics.completed}"
    )
    single_chunk = len(pieces) <= 1
    print(f"    → 조각 단위 도착 {'실패 (한 덩어리)' if single_chunk else '확인'}")

    # 2) 중간 취소 후 다음 호출
    stream = client.generate(PROMPT, max_tokens=256)
    first_piece = next(stream)
    stream.close()
    cancelled = client.last_metrics
    assert cancelled is not None
    after = list(client.generate(PROMPT, max_tokens=64))
    print(
        f"\n[2] 취소: 첫 조각 {first_piece!r} 받고 닫음 (완주 {cancelled.completed})"
        f" → 다음 호출 조각 {len(after)}개 정상"
    )

    # 3) 없는 모델
    ghost = _client("ghost-model:latest", thinking=False)
    try:
        list(ghost.generate("아무거나", max_tokens=8))
        print("\n[3] 없는 모델: 예외가 안 남 — 실패")
        return 1
    except GenerationError as exc:
        print(f"\n[3] 없는 모델: GenerationError 확인 ({str(exc)[:80]})")

    # 4) thinking 비교 (이 기기 값)
    print("\n[4] thinking 비교 (256토큰, 2회째 값)")
    for thinking in (False, True):
        timed = _client("qwen3.5:4b-q4_K_M", thinking=thinking)
        list(timed.generate(PROMPT, max_tokens=256))
        list(timed.generate(PROMPT, max_tokens=256))
        result = timed.last_metrics
        assert result is not None
        assert result.first_token_seconds is not None
        print(
            f"    thinking={'ON ' if thinking else 'OFF'}:"
            f" 첫 출력 {result.first_token_seconds:.2f}s / 총 {result.total_seconds:.2f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
