"""실제 Ollama로 임베딩 어댑터를 검증한다 (#16 완료 기준의 실물 항목).

단위 테스트가 못 보는 것 셋을 본다.

1. 실물 청크의 벡터 차원이 1024인가
2. max_input_tokens(num_batch)를 넘는 입력이 실제로 잘리는가 —
   전체 텍스트와 앞부분만 자른 텍스트의 코사인이 1.0에 붙으면 뒷부분이 버려진 것
3. 이 기기의 색인 처리량 (기기 종속 값이므로 기기명과 함께 기록)

사용:
    python scripts/verify_embedding_real.py --pdf <매뉴얼 PDF 하나>
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402
from techdoc_rag.ingestion.chunker import RecursiveChunker  # noqa: E402


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--pdf", type=Path, required=True)
    argument_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    arguments = argument_parser.parse_args()

    model = OllamaEmbeddingModel(
        model_name="bge-m3",
        endpoint=arguments.endpoint,
        batch_size=32,
        num_batch=2048,
        embedding_version="v1",
    )
    print(
        f"기기: {platform.node()} / 모델: {model.model_name}"
        f" / num_batch: {model.max_input_tokens}"
    )

    parsed = PypdfiumParser().parse(arguments.pdf)
    chunks = RecursiveChunker(size_chars=1200, overlap_chars=150, config_version="v1").chunk(
        parsed, "verify", 1
    )
    longest = max(chunks, key=lambda chunk: len(chunk.text))
    print(
        f"청크 {len(chunks)}개, 최장 {len(longest.text)}자"
        f" (p.{longest.page_start}~{longest.page_end})"
    )

    # 1) 차원
    vector = model.embed_query(longest.text)
    print(f"\n[1] 실물 청크 차원: {len(vector)} (기대 1024) → {'통과' if len(vector) == 1024 else '실패'}")

    # 2) 잘림 지점 — 청크를 이어붙여 num_batch를 확실히 넘긴 뒤 접두사와 비교
    long_text = longest.text * 6  # 최장 청크가 ~780토큰이므로 6배면 ~4,700토큰
    full = model.embed_query(long_text)
    print(f"\n[2] 잘림 확인 (전체 {len(long_text)}자 대 접두사, cosine)")
    truncated = False
    for ratio in (0.3, 0.5, 0.7, 0.9):
        prefix = long_text[: int(len(long_text) * ratio)]
        cosine = _cosine(full, model.embed_query(prefix))
        marker = ""
        if cosine > 0.9999:
            marker = "  ← 여기부터 전체와 동일: 이 뒤는 버려짐"
            truncated = True
        print(f"    접두사 {int(ratio * 100):>3}%: {cosine:.6f}{marker}")
    outcome = (
        "잘림 (계약대로 초과분 무시 확인됨)"
        if truncated
        else "잘리지 않는 것으로 관측됨 [확인 필요]"
    )
    print(f"    → num_batch({model.max_input_tokens}토큰) 초과분이 {outcome}")

    # 3) 처리량 — 실물 청크 32개 배치
    sample = [chunk.text for chunk in chunks[:32]]
    model.embed_documents(sample[:1])  # 워밍업(모델 적재) 제외
    started = time.perf_counter()
    model.embed_documents(sample)
    elapsed = time.perf_counter() - started
    print(f"\n[3] 처리량: 32청크 {elapsed:.1f}s = {32 / elapsed:.2f} chunk/s (이 기기 값. 기기 종속)")
    print(f"    2,454청크 환산: {2454 / (32 / elapsed) / 60:.0f}분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
