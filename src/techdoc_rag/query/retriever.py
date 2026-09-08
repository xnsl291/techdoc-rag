"""질문에서 근거 청크를 찾는다 (#19의 검색 단계).

질문을 벡터로 만들고, 활성 문서로만 한정해 검색한 뒤, 점수가
similarity_threshold 미만인 것을 버린다. threshold 0.0은 사실상 "거르지 않음"이다
— 코사인 점수는 이론상 음수가 가능해 0.0도 필터이긴 하나, 실물 텍스트에서
음수 유사도는 드물다 [추정]. 값 자체가 [미확정, 시작점]이며 평가셋 실험으로 정한다.

질의 확장이 붙으면 표현 여러 개로 검색해 결과를 합친다(#32). 합칠 때는
같은 청크의 가장 높은 점수를 남기고, 상위 top_k만 넘긴다.

활성 목록 조회(SQLite)와 벡터 검색(Qdrant) 사이에 버전이 전환되면 구버전
근거가 쓰일 수 있는 창이 있다. 단일 운영자·로컬 환경이라 전환과 질의가 겹칠
확률이 낮아 지금은 기록만 하고 완화하지 않는다(02 DP-55). FastAPI로 다중
사용자를 받는 시점에 재검토한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from techdoc_rag.domain.chunk import RetrievedChunk
from techdoc_rag.domain.errors import IndexingError, RetrievalError
from techdoc_rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore
from techdoc_rag.query.query_expander import QueryExpander

# 이웃 청크에 물려줄 점수 비율. 원래 걸린 청크보다 확실히 낮되, 다른 질의에서
# 걸린 청크들 사이의 순서는 흔들지 않을 만큼만 낮춘다.
_NEIGHBOR_SCORE_RATIO = 0.99


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """임계값을 통과한 청크와, 걸러진 개수.

    걸러진 개수를 같이 돌려주는 이유: "아무것도 못 찾음"(NO_RELEVANT_CHUNK)과
    "찾았으나 전부 관련도 미달"(LOW_RELEVANCE)은 개선 방향이 다른데,
    통과분만 보면 상위에서 이 둘을 구분할 수 없다.
    """

    chunks: list[RetrievedChunk]
    dropped_below_threshold: int


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        repository: DocumentRepository,
        top_k: int,
        similarity_threshold: float,
        query_expander: QueryExpander | None = None,
        expand_to_neighbors: bool = False,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._repository = repository
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        # 없으면 원문 한 번만 검색한다. 확장은 검색 전략이므로 여기에 둔다 —
        # 상위 계층은 "질문에 맞는 근거를 달라"고만 하고 방법은 모른다.
        self._query_expander = query_expander
        self._expand_to_neighbors = expand_to_neighbors

    def retrieve(
        self, question: str, document_ids: list[str] | None = None
    ) -> RetrievalResult:
        """질문과 관련된 청크를 점수 내림차순으로 돌려준다.

        빈 결과는 실패가 아니다 — 활성 문서가 없거나 관련 청크가 없는 것이고,
        No-answer 판단은 상위(chat_service)가 한다. 저장소 접근 실패는
        RetrievalError로 그대로 올라간다(D-005: 일반 지식으로 우회하지 않음).

        document_ids를 주면 그 문서로만 좁힌다. 문서 간 비교에 필요하다 —
        전체 검색은 점수가 높은 한 문서가 상위를 차지해 다른 문서가 밀린다
        (2026-09-08 실측: "정격 전류" 상위 6개가 전부 G100이었고 M100은 없었음).
        """
        active_ids = self._repository.active_document_ids()
        if document_ids is not None:
            # 문서를 지정하면 활성 목록과 교집합만 본다. 활성 검사를 건너뛰면
            # 색인은 끝났지만 아직 노출하지 않은 문서가 근거로 쓰인다.
            active_ids = [item for item in active_ids if item in set(document_ids)]
        if not active_ids:
            # 검색 대상이 없으면 질문 임베딩(수십 ms의 Ollama 호출)도 아낀다.
            return RetrievalResult(chunks=[], dropped_below_threshold=0)

        queries = (
            self._query_expander.expand(question) if self._query_expander else [question]
        )
        best: dict[str, RetrievedChunk] = {}
        dropped = 0
        for query in queries:
            try:
                query_vector = self._embedding_model.embed_query(query)
            except IndexingError as error:
                # 임베딩 어댑터는 색인 경로용이라 IndexingError를 던지지만, 질의
                # 임베딩 실패는 검색 실패다. 그대로 흘리면 HTTP 계층의 오류 매핑
                # (Retrieval/Generation→503)에 안 걸려 500이 난다 — 실물에서 발견.
                raise RetrievalError(f"질문 임베딩 실패: {error}") from error
            results = self._vector_store.search(
                query_vector, top_k=self._top_k, active_document_ids=active_ids
            )
            for result in results:
                if result.score < self._similarity_threshold:
                    dropped += 1
                    continue
                # 같은 청크가 여러 표현에서 나오면 가장 높은 점수를 남긴다.
                # 표현마다 점수가 다르므로 나중 것으로 덮으면 순서가 흔들린다.
                previous = best.get(result.chunk.chunk_id)
                if previous is None or result.score > previous.score:
                    best[result.chunk.chunk_id] = result

        # 표현을 늘린 만큼 후보도 늘어난다. 상위 top_k만 남겨 프롬프트 예산을 지킨다.
        chunks = sorted(best.values(), key=lambda item: item.score, reverse=True)[: self._top_k]
        if self._expand_to_neighbors:
            chunks = self._with_neighbors(chunks)
        return RetrievalResult(chunks=chunks, dropped_below_threshold=dropped)

    def _with_neighbors(self, found: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """찾은 청크와 페이지가 겹치는 이웃을 함께 넣는다.

        표가 청크 경계에서 잘리면 항목명과 값이 다른 조각에 남고, 검색은 둘 중
        하나만 물어 온다. 2026-09-08 실측에서 정격표가 0177/0178로 갈렸고 검색이
        값 없는 0178을 골라 "확인할 수 없습니다"가 나왔다.

        이웃은 검색으로 뽑힌 것이 아니므로 점수를 물려받되 조금 낮춘다. 원래
        걸린 청크가 먼저 오고 이웃이 뒤따라야, 예산이 모자랄 때 이웃부터 빠진다.
        """
        collected = {item.chunk.chunk_id: item for item in found}
        for item in found:
            chunk = item.chunk
            for neighbor in self._vector_store.fetch_overlapping(
                chunk.document_id, chunk.page_start, chunk.page_end
            ):
                if neighbor.chunk_id in collected:
                    continue
                collected[neighbor.chunk_id] = RetrievedChunk(
                    chunk=neighbor, score=item.score * _NEIGHBOR_SCORE_RATIO
                )
        return sorted(collected.values(), key=lambda item: item.score, reverse=True)
