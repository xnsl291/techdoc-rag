"""실물 매뉴얼 6건을 청킹해 규모와 분포를 기록한다.

수치는 기록만 하고 합격선으로 쓰지 않는다. 여기서 나온 청크 수가
Qdrant 용량 산정(05 §4.7)과 M8(chunk size 실험)의 출발 데이터가 된다.

사용:
    python scripts/record_chunking.py --manifest <manifest.json> --pdf-dir <PDF 폴더>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402
from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.ingestion.chunker import RecursiveChunker  # noqa: E402


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--manifest", type=Path, required=True)
    argument_parser.add_argument("--pdf-dir", type=Path, required=True)
    arguments = argument_parser.parse_args()

    settings = load_settings().chunking
    chunker = RecursiveChunker(
        size_chars=settings.size_chars,
        overlap_chars=settings.overlap_chars,
        config_version=settings.config_version,
    )
    pdf_parser = PypdfiumParser()
    print(
        f"설정: size_chars={settings.size_chars} overlap_chars={settings.overlap_chars} "
        f"({settings.config_version}, [미확정, 시작점])"
    )
    print()
    columns = ["document_id", "페이지", "청크", "중앙값", "최소", "최대", "초"]
    widths = [12, 6, 6, 6, 5, 5, 6]
    print(" ".join(f"{name:>{width}}" for name, width in zip(columns, widths, strict=True)))

    total_chunks = 0
    for entry in json.loads(arguments.manifest.read_text(encoding="utf-8")):
        pdf_path = arguments.pdf_dir / entry["file_name"]
        started_at = time.perf_counter()
        parsed = pdf_parser.parse(pdf_path)
        chunks = chunker.chunk(parsed, entry["document_id"], 1)
        elapsed_seconds = time.perf_counter() - started_at

        lengths = [len(chunk.text) for chunk in chunks]
        total_chunks += len(chunks)
        print(
            f"{entry['document_id']:<12} {parsed.page_count:>6} {len(chunks):>6} "
            f"{int(statistics.median(lengths)):>6} {min(lengths):>5} {max(lengths):>5} "
            f"{elapsed_seconds:>6.1f}"
        )

    print()
    print(f"총 청크 수: {total_chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
