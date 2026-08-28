"""파싱 결과인 페이지 단위 텍스트."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    """한 페이지에서 추출한 텍스트.

    has_text_layer가 False인 것은 스캔 이미지처럼 추출할 텍스트가 원래 없는 페이지다.
    추출 중 오류가 난 페이지는 여기 표시하지 않고 ParsedDocument.failed_pages에 남긴다.
    FR-002가 "텍스트가 없는 페이지"와 "파싱 실패"를 구분해 식별하라고 요구하기 때문이다.
    """

    page_no: int  # 1부터 시작. 원문 PDF의 페이지 번호와 같아야 Citation이 성립한다.
    text: str
    has_text_layer: bool


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """문서 하나의 파싱 결과.

    실패한 페이지도 pages에서 자리를 비우지 않는다. 빈 텍스트로 자리를 유지해야
    page_no가 원문과 어긋나지 않고, 어느 페이지가 실패했는지는 failed_pages로 안다.
    """

    pages: tuple[PageText, ...]
    failed_pages: tuple[int, ...] = ()

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_without_text_layer(self) -> int:
        return sum(1 for page in self.pages if not page.has_text_layer)
