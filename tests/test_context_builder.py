"""ContextBuilder 테스트 (#19)."""

from __future__ import annotations

import pytest

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.errors import ConfigurationError
from techdoc_rag.query.context_builder import ContextBuilder


def _retrieved(
    chunk_id: str, score: float, text: str = "본문", page_start: int = 10, page_end: int = 10
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="ls-m100-v1",
            document_version=1,
            page_start=page_start,
            page_end=page_end,
            text=text,
        ),
        score=score,
    )


NAMES = {"ls-m100-v1": "M100 사용설명서"}


def test_번호와_문서명_페이지가_붙어_조립된다() -> None:
    builder = ContextBuilder(budget_chars=1000)

    built = builder.build(
        [
            _retrieved("a", 0.9, "과전류 보호 내용", page_start=10, page_end=11),
            _retrieved("b", 0.8, "정격 전류 표", page_start=25, page_end=25),
        ],
        NAMES,
    )

    assert "[1] (M100 사용설명서 p.10~11)\n과전류 보호 내용" in built.text
    assert "[2] (M100 사용설명서 p.25)\n정격 전류 표" in built.text
    assert [(s.number, s.retrieved.chunk.chunk_id) for s in built.sources] == [(1, "a"), (2, "b")]


def test_예산을_넘으면_낮은_점수부터_통째로_빠진다() -> None:
    """자르지 않는다 — 잘린 본문은 [번호]가 가리키는 내용과 어긋난다."""
    # 블록 실제 길이(본문+헤더+구분자) 기준으로 첫째·셋째는 들어가고 둘째는 안 들어가는 예산
    builder = ContextBuilder(budget_chars=400)

    built = builder.build(
        [
            _retrieved("a", 0.9, "가" * 200),
            _retrieved("b", 0.8, "나" * 200),  # 안 들어감
            _retrieved("c", 0.7, "다" * 30),  # 남은 예산에 들어감 — 건너뛴 뒤에도 계속 본다
        ],
        NAMES,
    )

    ids = [s.retrieved.chunk.chunk_id for s in built.sources]
    assert ids == ["a", "c"]
    assert "나" not in built.text  # 잘려서라도 들어가면 안 된다


def test_중복_chunk_id는_첫_것만_남는다() -> None:
    builder = ContextBuilder(budget_chars=1000)

    built = builder.build([_retrieved("a", 0.9), _retrieved("a", 0.8)], NAMES)

    assert len(built.sources) == 1


def test_문서명이_없으면_document_id를_쓴다() -> None:
    builder = ContextBuilder(budget_chars=1000)

    built = builder.build([_retrieved("a", 0.9)], display_names={})

    assert "(ls-m100-v1 p.10)" in built.text


def test_빈_입력은_빈_결과다() -> None:
    builder = ContextBuilder(budget_chars=1000)

    built = builder.build([], NAMES)

    assert built.text == ""
    assert built.sources == []


def test_청크_하나도_안_들어가는_예산은_설정_오류다() -> None:
    """조용히 빈 근거로 돌리면 설정 실수가 No-answer로 둔갑한다."""
    builder = ContextBuilder(budget_chars=50)

    with pytest.raises(ConfigurationError, match="예산"):
        builder.build([_retrieved("a", 0.9, "가" * 500)], NAMES)
