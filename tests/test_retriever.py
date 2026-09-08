"""Retriever 테스트 (#19).

검색 백엔드는 가짜다 — 활성 한정과 threshold 필터가 retriever의 책임이고,
벡터 유사도 계산은 Qdrant 어댑터 테스트가 이미 덮는다.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.errors import IndexingError, RetrievalError
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


def test_질문_임베딩_실패는_RetrievalError로_변환된다() -> None:
    """임베딩 어댑터는 IndexingError를 던지지만 질의 경로에서는 검색 실패다.
    변환하지 않으면 HTTP 오류 매핑(503)에 안 걸려 500이 난다(실물 검증에서 발견)."""

    class BrokenEmbedding:
        def embed_query(self, text: str) -> list[float]:
            raise IndexingError("임베딩 서버 접속 실패")

    retriever = Retriever(
        embedding_model=BrokenEmbedding(),
        vector_store=FakeVectorStore([]),
        repository=FakeRepository(["ls-m100-v1"]),
        top_k=5,
        similarity_threshold=0.0,
    )

    with pytest.raises(RetrievalError, match="질문 임베딩 실패"):
        retriever.retrieve("질문")


class FakeExpander:
    def __init__(self, variants: list[str]) -> None:
        self._variants = variants
        self.calls: list[str] = []

    def expand(self, question: str) -> list[str]:
        self.calls.append(question)
        return self._variants


class RecordingEmbedding:
    """표현마다 다른 벡터를 만들어, 어떤 표현으로 검색했는지 추적한다."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text)), 0.0]


class PerQueryStore:
    """표현별로 다른 결과를 돌려주는 가짜 저장소."""

    def __init__(self, by_vector: dict[float, list[RetrievedChunk]]) -> None:
        self._by_vector = by_vector

    def search(self, query_vector, top_k, active_document_ids):
        return self._by_vector.get(query_vector[0], [])


def test_확장된_표현마다_검색하고_결과를_합친다() -> None:
    embedding = RecordingEmbedding()
    # 길이로 표현을 구분한다: "정격출력"(4) / "정격 출력 사양"(8)
    store = PerQueryStore(
        {4.0: [_retrieved("a", 0.7)], 8.0: [_retrieved("b", 0.6)]}
    )
    retriever = Retriever(
        embedding_model=embedding,
        vector_store=store,
        repository=FakeRepository(["ls-m100-v1"]),
        top_k=5,
        similarity_threshold=0.0,
        query_expander=FakeExpander(["정격출력", "정격 출력 사양"]),
    )

    result = retriever.retrieve("정격출력")

    assert embedding.queries == ["정격출력", "정격 출력 사양"]
    assert [r.chunk.chunk_id for r in result.chunks] == ["a", "b"]  # 점수 내림차순


def test_같은_청크는_가장_높은_점수로_남는다() -> None:
    """표현마다 점수가 다르다. 나중 것으로 덮으면 순서가 흔들린다.

    높은 점수를 **먼저** 오게 둔다. 낮은 점수를 먼저 두면 나중 것으로 덮는
    구현에서도 우연히 통과해 검증력이 없다(커밋 전 변형 확인에서 발견).
    """
    store = PerQueryStore({4.0: [_retrieved("a", 0.9)], 8.0: [_retrieved("a", 0.4)]})
    retriever = Retriever(
        embedding_model=RecordingEmbedding(),
        vector_store=store,
        repository=FakeRepository(["ls-m100-v1"]),
        top_k=5,
        similarity_threshold=0.0,
        query_expander=FakeExpander(["정격출력", "정격 출력 사양"]),
    )

    result = retriever.retrieve("정격출력")

    assert len(result.chunks) == 1
    assert result.chunks[0].score == 0.9


def test_합친_뒤에도_top_k를_넘지_않는다() -> None:
    """표현을 늘린 만큼 후보도 늘어난다. 그대로 넘기면 프롬프트 예산을 넘는다."""
    store = PerQueryStore(
        {
            4.0: [_retrieved("a", 0.9), _retrieved("b", 0.8)],
            8.0: [_retrieved("c", 0.7), _retrieved("d", 0.6)],
        }
    )
    retriever = Retriever(
        embedding_model=RecordingEmbedding(),
        vector_store=store,
        repository=FakeRepository(["ls-m100-v1"]),
        top_k=3,
        similarity_threshold=0.0,
        query_expander=FakeExpander(["정격출력", "정격 출력 사양"]),
    )

    result = retriever.retrieve("정격출력")

    assert [r.chunk.chunk_id for r in result.chunks] == ["a", "b", "c"]


def test_확장기가_없으면_원문만_검색한다() -> None:
    embedding = RecordingEmbedding()
    retriever = Retriever(
        embedding_model=embedding,
        vector_store=PerQueryStore({4.0: [_retrieved("a", 0.9)]}),
        repository=FakeRepository(["ls-m100-v1"]),
        top_k=5,
        similarity_threshold=0.0,
    )

    retriever.retrieve("정격출력")

    assert embedding.queries == ["정격출력"]
