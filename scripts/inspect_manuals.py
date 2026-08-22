"""수집한 매뉴얼의 본문 특성을 조사해 Manifest를 채운다.

`collect_manuals.py`가 API에서 얻을 수 있는 것만 기록하고 나머지는 비워 둔다.
여기서 PDF를 실제로 열어 페이지 수와 텍스트 레이어 유무를 채운다.

DP-07 관련: 파서로 pypdfium2를 쓴다. PyMuPDF는 AGPL-3.0이라 이 코드를 참고하거나
재사용하려는 쪽에 라이선스 부담을 넘기게 된다. 현재 필요한 것은 FR-002가 요구하는
페이지 단위 텍스트 추출과 텍스트 없는 페이지 식별이고 pypdfium2로 충분하다.
표 추출이 실제 병목으로 확인되면 그때 재검토한다.

사용법:
    python scripts/inspect_manuals.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pypdfium2

# 페이지에 이 글자 수 미만이면 텍스트 레이어가 없다고 본다.
# 스캔 페이지에도 머리말이나 쪽번호가 텍스트로 남는 경우가 있어 0으로 판정하면 놓친다.
MINIMUM_TEXT_LENGTH_PER_PAGE = 30

# 본문 특성을 볼 때 함께 세는 키워드. UC-1 질문이 어느 장에 몰리는지 가늠하는 용도다.
KEYWORD_LIST = ("트립", "과전류", "과전압", "정격", "파라미터", "고장", "점검", "경보")


@dataclass(frozen=True, slots=True)
class PdfInspection:
    """PDF 한 건을 열어 확인한 결과."""

    page_count: int
    pages_without_text_layer: int
    total_character_count: int
    keyword_page_counts: dict[str, int]

    @property
    def text_layer_ratio(self) -> float:
        if self.page_count == 0:
            return 0.0
        return round(1 - self.pages_without_text_layer / self.page_count, 3)


def inspect_pdf(pdf_path: Path) -> PdfInspection:
    """페이지별 텍스트를 추출해 특성을 센다.

    문서를 두 번 읽지 않도록 페이지를 한 번 순회하면서 필요한 값을 모두 모은다.
    140페이지 20MB짜리를 지표마다 다시 여는 것은 낭비다.
    """
    pages_without_text_layer = 0
    total_character_count = 0
    keyword_page_counts = dict.fromkeys(KEYWORD_LIST, 0)

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        for page in document:
            text = page.get_textpage().get_text_range()
            total_character_count += len(text)
            if len(text.strip()) < MINIMUM_TEXT_LENGTH_PER_PAGE:
                pages_without_text_layer += 1
            for keyword in KEYWORD_LIST:
                if keyword in text:
                    keyword_page_counts[keyword] += 1
        page_count = len(document)
    finally:
        document.close()

    return PdfInspection(
        page_count=page_count,
        pages_without_text_layer=pages_without_text_layer,
        total_character_count=total_character_count,
        keyword_page_counts=keyword_page_counts,
    )


def update_manifest(manifest: list[dict], documents_directory: Path) -> list[dict]:
    """Manifest의 빈 항목을 조사 결과로 채운다.

    model_name_from_content는 채우지 않는다. 파일명과 실제 모델이 다른 사례가 있어
    본문을 사람이 읽고 판단해야 한다. 자동으로 넣으면 틀린 값이 그대로 굳는다.
    """
    for entry in manifest:
        pdf_path = documents_directory / entry["document_id"] / "original.pdf"
        if not pdf_path.is_file():
            print(f"  건너뜀 {entry['document_id']}  파일 없음")
            continue

        inspection = inspect_pdf(pdf_path)
        entry.update(
            {
                "page_count": inspection.page_count,
                "pages_without_text_layer": inspection.pages_without_text_layer,
                "text_layer_ratio": inspection.text_layer_ratio,
                "total_character_count": inspection.total_character_count,
                "keyword_page_counts": inspection.keyword_page_counts,
            }
        )
        print(
            f"  {entry['document_id']:10s} {inspection.page_count:4d}p  "
            f"텍스트 없는 페이지 {inspection.pages_without_text_layer:3d}  "
            f"({inspection.text_layer_ratio:.1%} 추출 가능)"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="수집한 매뉴얼의 본문 특성 조사")
    parser.add_argument("--manifest-path", default="data/manifest.json")
    parser.add_argument("--documents-directory", default="data/documents")
    arguments = parser.parse_args()

    manifest_path = Path(arguments.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"{len(manifest)}건 조사\n")

    manifest = update_manifest(manifest, Path(arguments.documents_directory))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    inspected = [entry for entry in manifest if entry.get("page_count")]
    total_pages = sum(entry["page_count"] for entry in inspected)
    total_missing = sum(entry["pages_without_text_layer"] for entry in inspected)
    print(f"\n총 {total_pages}페이지, 텍스트 레이어 없는 페이지 {total_missing}개")
    print("실제 모델명은 본문을 읽고 직접 채울 것")


if __name__ == "__main__":
    main()
