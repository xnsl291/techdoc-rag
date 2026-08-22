"""검색 단위인 청크."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """검색 단위로 분할된 문서 조각.

    page_start와 page_end를 함께 두는 것은 청크가 페이지 경계를 넘을 수 있기 때문이다.
    Citation이 원문 위치를 가리키려면 이 범위가 보존되어야 한다.
    """

    chunk_id: str
    document_id: str
    document_version: int
    page_start: int
    page_end: int
    text: str
    section: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """검색 결과 한 건.

    점수는 Retrieval 방식에 따라 의미가 다르므로 서로 다른 방식의 점수를 직접 비교하지 않는다.
    """

    chunk: Chunk
    score: float
