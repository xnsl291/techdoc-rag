"""pypdfium2 기반 PDF 파서 어댑터.

이 모듈만 pypdfium2를 import한다. 서비스 계층은 domain.ports.PdfParser에만
의존하므로 파서 교체(DP-07 재검토 조건) 시 이 파일만 바꾸면 된다.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pypdfium2

from techdoc_rag.domain.errors import ParsingError
from techdoc_rag.domain.parsing import PageText, ParsedDocument

#: 공백 제거 후 이 길이 미만이면 텍스트 레이어가 없는 페이지로 본다.
#: scripts/inspect_manuals.py의 MINIMUM_TEXT_LENGTH_PER_PAGE와 같은 값이어야
#: Manifest(2026-08-22)와의 대조가 회귀 기준으로 성립한다.
MINIMUM_TEXT_LENGTH_PER_PAGE = 30


class PypdfiumParser:
    """pypdfium2로 페이지 단위 텍스트를 추출한다."""

    @property
    def parser_version(self) -> str:
        return f"pypdfium2-{importlib.metadata.version('pypdfium2')}"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        try:
            document = pypdfium2.PdfDocument(pdf_path)
        except Exception as exc:
            # pypdfium2는 안정된 예외 계층을 제공하지 않아 폭넓게 잡는다.
            # 파일 없음, 손상, 비PDF가 모두 여기로 온다.
            raise ParsingError(f"PDF를 열지 못함: {pdf_path} ({exc})") from exc

        pages: list[PageText] = []
        failed_pages: list[int] = []
        try:
            for index in range(len(document)):
                page_no = index + 1
                try:
                    text = document[index].get_textpage().get_text_range()
                except Exception:
                    # 실패한 페이지도 빈 텍스트로 자리를 유지한다.
                    # 빼 버리면 이후 페이지의 page_no가 원문과 어긋나 Citation이 깨진다.
                    failed_pages.append(page_no)
                    pages.append(PageText(page_no=page_no, text="", has_text_layer=False))
                    continue
                pages.append(
                    PageText(
                        page_no=page_no,
                        text=text,
                        has_text_layer=len(text.strip()) >= MINIMUM_TEXT_LENGTH_PER_PAGE,
                    )
                )
        finally:
            document.close()

        return ParsedDocument(pages=tuple(pages), failed_pages=tuple(failed_pages))
