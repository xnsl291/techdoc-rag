"""질문에서 근거 청크를 찾는다 (#19의 검색 단계).

질문을 벡터로 만들고, 활성 문서로만 한정해 검색한 뒤, 점수가
similarity_threshold 미만인 것을 버린다. threshold 0.0은 "거르지 않음"이다
(settings.yaml [미확정, 시작점] — 평가셋으로 실험하기 전까지 값이 없다).

활성 목록 조회(SQLite)와 벡터 검색(Qdrant) 사이에 버전이 전환되면 구버전
근거가 쓰일 수 있는 창이 있다. 단일 운영자·로컬 환경이라 전환과 질의가 겹칠
확률이 낮아 지금은 기록만 하고 완화하지 않는다(02 DP-55). FastAPI로 다중
사용자를 받는 시점에 재검토한다.
"""

from __future__ import annotations

from techdoc_rag.domain.chunk import RetrievedChunk
from techdoc_rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        repository: DocumentRepository,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._repository = repository
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """질문과 관련된 청크를 점수 내림차순으로 돌려준다.

        빈 결과는 실패가 아니다 — 활성 문서가 없거나 관련 청크가 없는 것이고,
        No-answer 판단은 상위(chat_service)가 한다. 저장소 접근 실패는
        RetrievalError로 그대로 올라간다(D-005: 일반 지식으로 우회하지 않음).
        """
        query_vector = self._embedding_model.embed_query(question)
        active_ids = self._repository.active_document_ids()
        results = self._vector_store.search(
            query_vector, top_k=self._top_k, active_document_ids=active_ids
        )
        return [
            result for result in results if result.score >= self._similarity_threshold
        ]
