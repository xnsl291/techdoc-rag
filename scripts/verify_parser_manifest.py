"""파서 어댑터의 결과를 Manifest와 대조한다.

단위 테스트는 손으로 만든 최소 PDF만 다루므로, 실물 매뉴얼 6건(3,021페이지)에서
PypdfiumParser가 inspect_manuals.py(2026-08-22)와 같은 값을 내는지 여기서 확인한다.
같은 라이브러리, 같은 판정 기준이므로 페이지 수와 텍스트 없는 페이지 수가
정확히 일치해야 한다. 불일치는 어댑터의 회귀다.

사용:
    python scripts/verify_parser_manifest.py --manifest <manifest.json> --pdf-dir <PDF 폴더>

처리 시간은 기록만 하고 합격선으로 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--manifest", type=Path, required=True)
    argument_parser.add_argument("--pdf-dir", type=Path, required=True)
    arguments = argument_parser.parse_args()

    manifest_entries = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    pdf_parser = PypdfiumParser()
    print(f"parser: {pdf_parser.parser_version}")
    print()
    print(f"{'document_id':<12} {'페이지':>6} {'무텍스트':>8} {'실패':>4} {'초':>6}  판정")

    mismatches = []
    for entry in manifest_entries:
        pdf_path = arguments.pdf_dir / entry["file_name"]
        if not pdf_path.exists():
            mismatches.append(f"{entry['document_id']}: 파일 없음 ({pdf_path.name})")
            print(f"{entry['document_id']:<12} {'—':>6} {'—':>8} {'—':>4} {'—':>6}  파일 없음")
            continue

        started_at = time.perf_counter()
        parsed = pdf_parser.parse(pdf_path)
        elapsed_seconds = time.perf_counter() - started_at

        problems = []
        if parsed.page_count != entry["page_count"]:
            problems.append(f"페이지 수 {parsed.page_count} != {entry['page_count']}")
        if parsed.pages_without_text_layer != entry["pages_without_text_layer"]:
            problems.append(
                f"무텍스트 {parsed.pages_without_text_layer}"
                f" != {entry['pages_without_text_layer']}"
            )
        if parsed.failed_pages:
            problems.append(f"실패 페이지 {parsed.failed_pages}")

        verdict = "일치" if not problems else "; ".join(problems)
        if problems:
            mismatches.append(f"{entry['document_id']}: {verdict}")
        print(
            f"{entry['document_id']:<12} {parsed.page_count:>6} "
            f"{parsed.pages_without_text_layer:>8} {len(parsed.failed_pages):>4} "
            f"{elapsed_seconds:>6.1f}  {verdict}"
        )

    print()
    if mismatches:
        print(f"불일치 {len(mismatches)}건:")
        for line in mismatches:
            print(f"  - {line}")
        return 1
    print(f"전체 일치 ({len(manifest_entries)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
