"""Retriever 테스트 (#19).

검색 백엔드는 가짜다 — 활성 한정과 threshold 필터가 retriever의 책임이고,
벡터 유사도 계산은 Qdrant 어댑터 테스트가 이미 덮는다.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.errors import RetrievalError
from techdoc_rag.query.retriever import Retriever


def _retrieved(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="ls-m100-v1",
            document_version=1,
            page_start=10,
            page_end=11,
            text=f"{chunk_id} 본문",
        ),
        score=score,
    )


class FakeEmbedding:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, active: list[str]) -> None:
        self._active = active

    def active_document_ids(self) -> list[str]:
        return self._active


class FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[int, Sequence[str]]] = []
        self.fail = False

    def search(
        self, query_vector: Sequence[float], top_k: int, active_document_ids: Sequence[str]
    ) -> list[RetrievedChunk]:
        if self.fail:
            raise RetrievalError("저장소 접근 불가")
        self.calls.append((top_k, active_document_ids))
        if not active_document_ids:
            return []
        return self._results


def _retriever(store: FakeVectorStore, active: list[str], threshold: float = 0.0) -> Retriever:
    return Retriever(
        embedding_model=FakeEmbedding(),
        vector_store=store,
        repository=FakeRepository(active),
        top_k=5,
        similarity_threshold=threshold,
    )


def test_활성_문서_목록이_검색_필터로_전달된다() -> None:
    store = FakeVectorStore([_retrieved("a", 0.9)])
    retriever = _retriever(store, active=["ls-m100-v1", "ls-g100-v1"])

    result = retriever.retrieve("정격 전류는?")

    assert [r.chunk.chunk_id for r in result.chunks] == ["a"]
    assert store.calls == [(5, ["ls-m100-v1", "ls-g100-v1"])]


def test_임계값_미만은_버려진다() -> None:
    store = FakeVectorStore([_retrieved("a", 0.9), _retrieved("b", 0.5), _retrieved("c", 0.4)])
    retriever = _retriever(store, active=["ls-m100-v1"], threshold=0.5)

    result = retriever.retrieve("질문")

    # 경계값 0.5는 남는다 — 미만만 버린다.
    assert [r.chunk.chunk_id for r in result.chunks] == ["a", "b"]
    assert result.dropped_below_threshold == 1


def test_임계값_0은_거르지_않는다() -> None:
    store = FakeVectorStore([_retrieved("a", 0.01)])
    retriever = _retriever(store, active=["ls-m100-v1"], threshold=0.0)

    assert len(retriever.retrieve("질문").chunks) == 1


def test_활성_문서가_없으면_빈_결과다() -> None:
    store = FakeVectorStore([_retrieved("a", 0.9)])
    retriever = _retriever(store, active=[])

    result = retriever.retrieve("질문")

    assert result.chunks == []
    assert result.dropped_below_threshold == 0


def test_저장소_실패는_RetrievalError로_올라간다() -> None:
    """검색 실패를 삼키고 빈 결과로 바꾸면 장애가 No-answer로 둔갑한다."""
    store = FakeVectorStore([])
    store.fail = True
    retriever = _retriever(store, active=["ls-m100-v1"])

    with pytest.raises(RetrievalError):
        retriever.retrieve("질문")
