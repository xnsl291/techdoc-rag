"""RecursiveChunker 테스트."""

from __future__ import annotations

import pytest

from techdoc_rag.domain.parsing import PageText, ParsedDocument
from techdoc_rag.ingestion.chunker import RecursiveChunker


def _document(*page_texts: str) -> ParsedDocument:
    pages = tuple(
        PageText(page_no=index + 1, text=text, has_text_layer=bool(text.strip()))
        for index, text in enumerate(page_texts)
    )
    return ParsedDocument(pages=pages)


def _chunker(size: int = 100, overlap: int = 20) -> RecursiveChunker:
    return RecursiveChunker(size_chars=size, overlap_chars=overlap, config_version="v1")


def test_같은_입력이면_결과가_완전히_같다() -> None:
    document = _document("문단 하나.\n\n" * 30, "둘째 페이지 문장. " * 40)

    first = _chunker().chunk(document, "doc-1", 1)
    second = _chunker().chunk(document, "doc-1", 1)

    assert first == second
    assert len(first) > 1


def test_chunk_id는_문서ID와_순번으로_결정된다() -> None:
    chunks = _chunker().chunk(_document("가나다 " * 200), "doc-1", 1)

    assert [chunk.chunk_id for chunk in chunks[:3]] == ["doc-1:0000", "doc-1:0001", "doc-1:0002"]
    assert all(chunk.document_id == "doc-1" and chunk.document_version == 1 for chunk in chunks)


def test_크기는_size와_overlap의_합을_넘지_않는다() -> None:
    size, overlap = 100, 20
    chunks = _chunker(size, overlap).chunk(_document("단어 " * 500), "doc-1", 1)

    assert all(len(chunk.text) <= size + overlap for chunk in chunks)


def test_연속_청크는_겹침을_가진다() -> None:
    # 공백으로만 나뉘는 텍스트라 청크 경계가 겹침 구간을 반드시 포함한다.
    chunks = _chunker(100, 20).chunk(_document("word " * 300), "doc-1", 1)

    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        head = current.text[:10]
        assert head in previous.text


def test_페이지_경계를_넘는_청크는_범위를_보존한다() -> None:
    # 페이지1 끝과 페이지2 시작이 한 청크에 묶이도록 페이지를 작게 만든다.
    chunks = _chunker(200, 0).chunk(_document("일페이지 " * 10, "이페이지 " * 10), "doc-1", 1)

    spanning = [chunk for chunk in chunks if chunk.page_start != chunk.page_end]
    assert spanning, "페이지를 넘는 청크가 있어야 하는 구성임"
    assert spanning[0].page_start == 1
    assert spanning[0].page_end == 2
    assert "일페이지" in spanning[0].text and "이페이지" in spanning[0].text


def test_한_페이지_안의_청크는_그_페이지_번호를_가진다() -> None:
    long_second_page = "둘째 페이지 문장. " * 100
    chunks = _chunker(150, 20).chunk(_document("첫 페이지 짧음.", long_second_page), "doc-1", 1)

    tail_chunks = [chunk for chunk in chunks if "둘째" in chunk.text and "첫" not in chunk.text]
    assert tail_chunks
    assert all(chunk.page_start == 2 and chunk.page_end == 2 for chunk in tail_chunks)


def test_빈_문서는_빈_목록이다() -> None:
    assert _chunker().chunk(_document("", "   "), "doc-1", 1) == []
    assert _chunker().chunk(ParsedDocument(pages=()), "doc-1", 1) == []


def test_구분자가_없는_긴_텍스트도_크기대로_잘린다() -> None:
    # 공백조차 없는 최악의 입력. 마지막 수단인 하드 컷이 동작해야 한다.
    chunks = _chunker(100, 0).chunk(_document("가" * 1000), "doc-1", 1)

    assert len(chunks) >= 10
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_잘못된_설정은_생성_시점에_거부한다() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(size_chars=0, overlap_chars=0, config_version="v1")
    with pytest.raises(ValueError):
        RecursiveChunker(size_chars=100, overlap_chars=100, config_version="v1")
