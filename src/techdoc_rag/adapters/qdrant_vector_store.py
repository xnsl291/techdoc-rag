"""Qdrant 벡터 저장소.

검색 인덱스일 뿐 정본이 아니다(D-004). 어떤 문서가 유효한지는 SQLite가 정하고,
여기는 그 목록을 필터로 받아 쓴다. 벡터와 청크는 원본 PDF에서 언제든 다시 만들 수 있다.

포인트 ID를 chunk_id에서 결정론적으로 만든다. 같은 문서를 다시 색인해도
같은 자리에 덮어써지므로 중복 벡터가 쌓이지 않는다(FR-004).

접속은 만들지 않고 주입받는다. 테스트는 로컬 모드, 운영은 서버 클라이언트를 넘긴다.
어댑터가 접속까지 책임지면 테스트에 서버가 필요해진다.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.errors import IndexingError, RetrievalError

# 포인트 ID 생성용 고정 네임스페이스. 이 값이 바뀌면 기존 벡터를 덮어쓰지 못하고
# 전부 새 포인트로 쌓이므로 절대 바꾸지 않는다.
POINT_ID_NAMESPACE = uuid.UUID("6f2b1e3c-7a94-4f0d-9c1b-2d5e8a3f7b60")

# payload 키. Chunk의 필드명을 그대로 쓴다. 저장소마다 이름이 달라지면
# 어느 쪽 표기가 맞는지 매번 확인해야 한다.
_PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "document_version",
    "page_start",
    "page_end",
    "text",
    "section",
)

# 한 번에 보낼 포인트 수. 실물이 6건 3,021페이지에서 2,454청크이고 본문이 1,200자 안팎이라
# 전량을 한 요청에 담으면 수십 MB가 된다. 서버에는 요청 크기 제한이 있어 거부한다.
UPSERT_BATCH_SIZE = 128


class QdrantVectorStore:
    """VectorStore Protocol 구현."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance

    def initialize(self) -> None:
        """컬렉션이 없으면 만들고, 이미 있으면 벡터 구성이 맞는지 본다.

        임베딩 모델을 바꾸면 벡터 차원이 달라진다. 그런데 같은 이름으로 붙으면
        여기서는 에러 없이 넘어가고, 색인을 한참 돌린 뒤에야 저장할 때마다 실패한다.

        document_id에 인덱스를 건다. 활성 문서 필터가 모든 검색에 붙으므로
        인덱스가 없으면 문서가 늘수록 전수 검사가 된다.
        """
        try:
            if self._client.collection_exists(self._collection_name):
                self._verify_existing_collection()
            else:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._vector_size, distance=self._distance
                    ),
                )
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except IndexingError:
            raise
        except Exception as error:
            raise IndexingError(f"컬렉션 준비 실패: {error}") from error

    def _verify_existing_collection(self) -> None:
        existing = self._client.get_collection(self._collection_name).config.params.vectors
        if not isinstance(existing, models.VectorParams):
            raise IndexingError(
                f"컬렉션 {self._collection_name}이 이름 있는 벡터 구성이라 쓸 수 없음"
            )
        if existing.size != self._vector_size or existing.distance != self._distance:
            raise IndexingError(
                f"컬렉션 {self._collection_name}의 벡터 구성이 다름: "
                f"기존 {existing.size}차원 {existing.distance}, "
                f"요청 {self._vector_size}차원 {self._distance}. "
                f"임베딩 모델을 바꿨다면 새 컬렉션으로 재색인할 것"
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise IndexingError(
                f"청크 {len(chunks)}개와 벡터 {len(vectors)}개의 수가 다름"
            )
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=list(vector),
                payload={field: getattr(chunk, field) for field in _PAYLOAD_FIELDS},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # 중간에 끊겨도 다시 실행하면 된다. 포인트 ID가 결정론적이라 이미 넣은 것은 덮어써진다.
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            try:
                self._client.upsert(collection_name=self._collection_name, points=batch)
            except Exception as error:
                raise IndexingError(
                    f"벡터 저장 실패 ({start}번째부터 {len(batch)}개): {error}"
                ) from error

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        active_document_ids: Sequence[str],
    ) -> list[RetrievedChunk]:
        """활성 문서로 한정해 유사한 청크를 찾는다.

        active 목록이 비면 질의 자체를 하지 않는다. 빈 필터를 넘기면 Qdrant는
        조건 없는 검색이 되어 비활성 문서까지 결과에 들어온다. 색인은 끝났지만
        아직 활성화하지 않은 문서가 답변 근거로 쓰이게 된다.
        """
        if not active_document_ids:
            return []

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=list(query_vector),
                limit=top_k,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchAny(any=list(active_document_ids)),
                        )
                    ]
                ),
            )
        except Exception as error:
            raise RetrievalError(f"벡터 검색 실패: {error}") from error

        return [self._to_retrieved_chunk(point) for point in response.points]

    def delete_document(self, document_id: str) -> None:
        """한 문서의 벡터를 모두 지운다. 없는 문서를 지워도 오류가 아니다."""
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
            )
        except Exception as error:
            raise IndexingError(f"문서 벡터 삭제 실패: {error}") from error

    def count(self) -> int:
        """저장된 포인트 수. 멱등성 검증과 운영 점검에 쓴다."""
        try:
            return self._client.count(self._collection_name, exact=True).count
        except Exception as error:
            raise RetrievalError(f"벡터 수 조회 실패: {error}") from error

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """chunk_id에서 포인트 ID를 만든다.

        Qdrant의 포인트 ID는 정수나 UUID만 허용하므로 chunk_id를 그대로 못 쓴다.
        uuid5는 같은 입력에서 항상 같은 값을 주므로 재색인이 덮어쓰기가 된다.
        """
        return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))

    @staticmethod
    def _to_retrieved_chunk(point: models.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            document_version=payload["document_version"],
            page_start=payload["page_start"],
            page_end=payload["page_end"],
            text=payload["text"],
            section=payload.get("section"),
        )
        return RetrievedChunk(chunk=chunk, score=point.score)
