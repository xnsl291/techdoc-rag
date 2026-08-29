"""QdrantVectorStore 테스트.

검증 대상은 저장·조회가 되는지가 아니라 계약이다.
같은 청크를 다시 넣어도 늘지 않는가(FR-004), 비활성 문서가 결과에 섞이지 않는가,
active 목록이 비었을 때 전체를 돌려주지는 않는가.

로컬 모드(`:memory:`)를 쓰므로 서버가 필요 없다.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from techdoc_rag.adapters.qdrant_vector_store import UPSERT_BATCH_SIZE, QdrantVectorStore
from techdoc_rag.domain.chunk import Chunk
from techdoc_rag.domain.errors import IndexingError, RetrievalError

VECTOR_SIZE = 4


def _chunk(chunk_id: str, document_id: str = "doc-a", page_start: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=1,
        page_start=page_start,
        page_end=page_start,
        text=f"{chunk_id} 본문",
        section="3.1 배선",
    )


def _vector(seed: float) -> list[float]:
    return [seed, 0.0, 0.0, 0.0]


@pytest.fixture
def store() -> QdrantVectorStore:
    vector_store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="test",
        vector_size=VECTOR_SIZE,
    )
    vector_store.initialize()
    return vector_store


def test_upsert_후_검색으로_찾을_수_있다(store: QdrantVectorStore) -> None:
    chunk = _chunk("doc-a:0001")
    store.upsert([chunk], [_vector(1.0)])

    results = store.search(_vector(1.0), top_k=5, active_document_ids=["doc-a"])

    assert len(results) == 1
    found = results[0].chunk
    assert found.chunk_id == chunk.chunk_id
    assert found.document_id == chunk.document_id
    assert found.page_start == chunk.page_start
    assert found.text == chunk.text
    assert found.section == chunk.section


def test_같은_청크를_두_번_넣어도_늘지_않는다(store: QdrantVectorStore) -> None:
    """FR-004. 포인트 ID가 chunk_id에서 결정론적으로 나오므로 재실행은 덮어쓰기가 된다."""
    chunk = _chunk("doc-a:0001")
    store.upsert([chunk], [_vector(1.0)])
    store.upsert([chunk], [_vector(1.0)])

    assert store.count() == 1


def test_같은_chunk_id로_본문이_바뀌면_덮어쓴다(store: QdrantVectorStore) -> None:
    original = _chunk("doc-a:0001")
    store.upsert([original], [_vector(1.0)])

    revised = Chunk(
        chunk_id=original.chunk_id,
        document_id=original.document_id,
        document_version=2,
        page_start=7,
        page_end=8,
        text="고친 본문",
        section=None,
    )
    store.upsert([revised], [_vector(1.0)])

    results = store.search(_vector(1.0), top_k=5, active_document_ids=["doc-a"])
    assert store.count() == 1
    assert results[0].chunk.text == "고친 본문"
    assert results[0].chunk.document_version == 2
    assert results[0].chunk.section is None


def test_active_목록에_없는_문서는_검색되지_않는다(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("doc-a:0001", document_id="doc-a")], [_vector(1.0)])
    store.upsert([_chunk("doc-b:0001", document_id="doc-b")], [_vector(1.0)])

    results = store.search(_vector(1.0), top_k=10, active_document_ids=["doc-a"])

    assert [r.chunk.document_id for r in results] == ["doc-a"]


def test_active_목록이_비면_빈_결과를_돌려준다(store: QdrantVectorStore) -> None:
    """빈 필터로 질의하면 전체가 조회된다. 그 상황을 막으려고 둔 테스트다."""
    store.upsert([_chunk("doc-a:0001")], [_vector(1.0)])

    assert store.search(_vector(1.0), top_k=10, active_document_ids=[]) == []


def test_delete_document는_다른_문서를_건드리지_않는다(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("doc-a:0001", document_id="doc-a")], [_vector(1.0)])
    store.upsert([_chunk("doc-b:0001", document_id="doc-b")], [_vector(1.0)])

    store.delete_document("doc-a")

    results = store.search(_vector(1.0), top_k=10, active_document_ids=["doc-a", "doc-b"])
    assert [r.chunk.document_id for r in results] == ["doc-b"]
    assert store.count() == 1


def test_없는_문서를_지워도_오류가_아니다(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("doc-a:0001")], [_vector(1.0)])
    store.delete_document("doc-none")
    assert store.count() == 1


def test_빈_청크_목록은_아무_일도_하지_않는다(store: QdrantVectorStore) -> None:
    store.upsert([], [])
    assert store.count() == 0


def test_청크와_벡터_개수가_다르면_거부한다(store: QdrantVectorStore) -> None:
    with pytest.raises(IndexingError):
        store.upsert([_chunk("doc-a:0001")], [])


def test_벡터_차원이_맞지_않으면_IndexingError(store: QdrantVectorStore) -> None:
    with pytest.raises(IndexingError):
        store.upsert([_chunk("doc-a:0001")], [[1.0, 2.0]])


def test_배치_크기를_넘겨도_전부_저장된다(store: QdrantVectorStore) -> None:
    """실물은 2,454청크다. 한 요청에 다 담으면 서버가 거부한다."""
    count = UPSERT_BATCH_SIZE * 2 + 5
    chunks = [_chunk(f"doc-a:{index:04d}", page_start=index + 1) for index in range(count)]
    store.upsert(chunks, [_vector(1.0) for _ in range(count)])

    assert store.count() == count


def test_top_k만큼만_돌려준다(store: QdrantVectorStore) -> None:
    chunks = [_chunk(f"doc-a:{index:04d}", page_start=index + 1) for index in range(5)]
    store.upsert(chunks, [_vector(1.0 - index * 0.1) for index in range(5)])

    results = store.search(_vector(1.0), top_k=2, active_document_ids=["doc-a"])

    assert len(results) == 2


def test_검색_결과가_없어도_예외가_아니다(store: QdrantVectorStore) -> None:
    """0건은 실패가 아니라 No-answer의 입력이다."""
    assert store.search(_vector(1.0), top_k=5, active_document_ids=["doc-a"]) == []


def test_컬렉션이_없으면_검색은_RetrievalError() -> None:
    """initialize를 부르지 않은 상태. 조회 실패는 Retrieval 축으로 구분한다."""
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="absent",
        vector_size=VECTOR_SIZE,
    )
    with pytest.raises(RetrievalError):
        store.search(_vector(1.0), top_k=5, active_document_ids=["doc-a"])


def test_initialize는_여러_번_불러도_안전하다(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("doc-a:0001")], [_vector(1.0)])
    store.initialize()
    assert store.count() == 1


def test_기존_컬렉션의_벡터_차원이_다르면_거부한다() -> None:
    """임베딩 모델을 바꾸고 같은 컬렉션에 붙는 상황. 색인 도중이 아니라 시작 시점에 끊는다."""
    client = QdrantClient(location=":memory:")
    QdrantVectorStore(client=client, collection_name="test", vector_size=4).initialize()

    other = QdrantVectorStore(client=client, collection_name="test", vector_size=768)
    with pytest.raises(IndexingError) as error:
        other.initialize()

    assert "재색인" in str(error.value)
