"""PypdfiumParser 테스트.

실제 매뉴얼 6건(94MB)은 저장소에 없으므로 여기서는 손으로 만든 최소 PDF로
계약을 검증한다. 실물 대조는 scripts/verify_parser_manifest.py가 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from techdoc_rag.adapters import pypdfium_parser
from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser
from techdoc_rag.domain.errors import ParsingError

# 30자(공백 제거 후) 이상이어야 텍스트 레이어가 있다고 판정된다.
LONG_TEXT = "TechdocRagParserFixtureText0123456789ABCDEF"


def _build_minimal_pdf(page_texts: list[str | None]) -> bytes:
    """페이지별 텍스트를 가진 최소 PDF를 손으로 조립한다.

    None인 페이지는 콘텐츠 스트림 없이 만들어 텍스트 레이어가 없는 페이지가 된다.
    xref 오프셋을 실제 바이트 위치로 계산하므로 pdfium의 파싱을 통과한다.
    """
    objects: list[bytes] = []
    page_object_numbers = []
    next_object_number = 4  # 1=catalog, 2=pages, 3=font

    page_entries = []
    for text in page_texts:
        page_number = next_object_number
        next_object_number += 1
        if text is None:
            page_entries.append((page_number, None, None))
        else:
            content_number = next_object_number
            next_object_number += 1
            page_entries.append((page_number, content_number, text))
        page_object_numbers.append(page_number)

    kids = " ".join(f"{n} 0 R" for n in page_object_numbers)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    page_total = len(page_object_numbers)
    objects.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {page_total} >>\nendobj\n".encode()
    )
    objects.append(
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    for page_number, content_number, text in page_entries:
        if content_number is None:
            objects.append(
                f"{page_number} 0 obj\n<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 612 792] >>\nendobj\n".encode()
            )
        else:
            objects.append(
                f"{page_number} 0 obj\n<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 612 792] /Contents {content_number} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n".encode()
            )
            stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
            objects.append(
                f"{content_number} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
                + stream
                + b"\nendstream\nendobj\n"
            )

    header = b"%PDF-1.4\n"
    body = b""
    offsets = []
    for obj in objects:
        offsets.append(len(header) + len(body))
        body += obj

    xref_position = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n".encode()
    )
    return header + body + xref + trailer


@pytest.fixture()
def mixed_pdf(tmp_path: Path) -> Path:
    """1페이지는 텍스트 있음, 2페이지는 텍스트 레이어 없음."""
    path = tmp_path / "mixed.pdf"
    path.write_bytes(_build_minimal_pdf([LONG_TEXT, None]))
    return path


def test_텍스트_페이지와_빈_페이지를_구분한다(mixed_pdf: Path) -> None:
    result = PypdfiumParser().parse(mixed_pdf)

    assert result.page_count == 2
    assert result.failed_pages == ()
    assert [page.page_no for page in result.pages] == [1, 2]
    assert result.pages[0].has_text_layer is True
    assert LONG_TEXT in result.pages[0].text
    assert result.pages[1].has_text_layer is False
    assert result.pages_without_text_layer == 1


def test_짧은_텍스트는_텍스트_레이어_없음으로_판정한다(tmp_path: Path) -> None:
    # 30자 미만이면 inspect_manuals.py와 같은 기준으로 텍스트 없는 페이지다.
    path = tmp_path / "short.pdf"
    path.write_bytes(_build_minimal_pdf(["short"]))

    result = PypdfiumParser().parse(path)

    assert result.pages[0].has_text_layer is False


def test_없는_파일은_ParsingError(tmp_path: Path) -> None:
    with pytest.raises(ParsingError):
        PypdfiumParser().parse(tmp_path / "missing.pdf")


def test_PDF가_아닌_파일은_ParsingError(tmp_path: Path) -> None:
    path = tmp_path / "not_a_pdf.pdf"
    path.write_bytes(b"this is not a pdf at all")

    with pytest.raises(ParsingError):
        PypdfiumParser().parse(path)


def test_개별_페이지_실패는_자리를_유지하고_failed_pages에_남는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2페이지 추출만 실패하는 상황을 가짜 문서로 재현한다.

    실물 PDF로는 페이지 단위 실패를 재현하기 어려워 pypdfium2 경계만 바꿔치기한다.
    검증 대상은 pypdfium2가 아니라 우리 쪽 실패 처리 로직이다.
    """

    class FakeTextPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text_range(self) -> str:
            return self._text

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def get_textpage(self) -> FakeTextPage:
            if self._text is None:
                raise RuntimeError("extraction failed")
            return FakeTextPage(self._text)

    class FakeDocument:
        def __init__(self, _path: object) -> None:
            self._pages = [FakePage(LONG_TEXT), FakePage(None), FakePage(LONG_TEXT)]

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, index: int) -> FakePage:
            return self._pages[index]

        def close(self) -> None:
            pass

    monkeypatch.setattr(pypdfium_parser.pypdfium2, "PdfDocument", FakeDocument)

    result = PypdfiumParser().parse(Path("irrelevant.pdf"))

    assert result.page_count == 3
    assert result.failed_pages == (2,)
    assert [page.page_no for page in result.pages] == [1, 2, 3]
    assert result.pages[1].text == ""
    # 실패 페이지는 has_text_layer=False로도 세어지지만 failed_pages로 구분된다.
    assert result.pages_without_text_layer == 1


def test_parser_version은_라이브러리와_버전을_담는다() -> None:
    version = PypdfiumParser().parser_version

    assert version.startswith("pypdfium2-")
    assert version != "pypdfium2-"
