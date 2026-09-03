"""질문 하나를 답과 Citation으로 바꾼다 (#19의 마지막 단계).

검색(retriever)과 조립(context_builder)과 LLM을 엮는 자리다.

No-answer 세 갈래(01 §18, 05 §4.3):
- NO_RELEVANT_CHUNK — 검색이 아무것도 못 찾음 (활성 문서 없음 포함)
- LOW_RELEVANCE — 찾았으나 전부 임계값 미만
- NOT_GROUNDED — 답을 만들었으나 근거 사용을 확인할 수 없음

근거 확인(v1)은 인용 번호 방식이다: 프롬프트로 [1] 표기를 강제하고 답 속의
번호를 파싱해 is_used_in_answer를 채운다. 번호가 하나도 없으면 근거를 썼다는
증거가 없으므로 NOT_GROUNDED다. 2차 LLM 검증은 응답 시간이 2배라 채택하지
않았다(승인 2026-09-03). 4B가 표기를 실제로 따르는지는 실물 스크립트로 확인한다.

검색 실패(RetrievalError)와 생성 실패(GenerationError)는 여기서 잡지 않는다.
장애를 No-answer로 감추면 품질 문제와 장애를 구분할 수 없게 된다(D-005).
"""

from __future__ import annotations

import re

from techdoc_rag.domain.answer import Answer, Citation, NoAnswerReason
from techdoc_rag.domain.chunk import RetrievedChunk
from techdoc_rag.domain.ports import DocumentRepository, LlmClient
from techdoc_rag.query.context_builder import BuiltContext, ContextBuilder
from techdoc_rag.query.retriever import Retriever

# 근거에 없는 내용을 답하지 않는 것과 [번호] 표기가 계약의 전부다.
# 바꾸면 NOT_GROUNDED 판정 분포가 통째로 바뀌므로 버전을 올리고 기록한다.
PROMPT_VERSION = "v1"

_PROMPT_TEMPLATE = """다음 근거만 사용해 질문에 답하라. 근거에 없는 내용은 답하지 마라.
답변에 사용한 근거의 번호를 문장 끝에 [1] 형태로 반드시 표기하라.
근거에서 답을 찾을 수 없으면 "근거 자료에서 확인할 수 없습니다"라고만 답하라.

[근거]
{context}

[질문]
{question}

[답변]
"""

_CITATION_MARK = re.compile(r"\[(\d+)\]")


class ChatService:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm_client: LlmClient,
        repository: DocumentRepository,
        max_answer_tokens: int,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm_client = llm_client
        self._repository = repository
        self._max_answer_tokens = max_answer_tokens

    def ask(self, question: str) -> Answer:
        retrieval = self._retriever.retrieve(question)
        if not retrieval.chunks:
            reason = (
                NoAnswerReason.LOW_RELEVANCE
                if retrieval.dropped_below_threshold > 0
                else NoAnswerReason.NO_RELEVANT_CHUNK
            )
            return Answer(
                text="등록된 문서에서 관련 근거를 찾지 못했습니다.", no_answer_reason=reason
            )

        display_names = self._display_names(retrieval.chunks)
        context = self._context_builder.build(retrieval.chunks, display_names)
        prompt = _PROMPT_TEMPLATE.format(context=context.text, question=question)
        text = "".join(self._llm_client.generate(prompt, max_tokens=self._max_answer_tokens))

        cited = self._cited_numbers(text, context)
        citations = [
            Citation(
                document_id=source.retrieved.chunk.document_id,
                document_version=source.retrieved.chunk.document_version,
                display_name=display_names.get(
                    source.retrieved.chunk.document_id, source.retrieved.chunk.document_id
                ),
                page_start=source.retrieved.chunk.page_start,
                page_end=source.retrieved.chunk.page_end,
                chunk_id=source.retrieved.chunk.chunk_id,
                is_used_in_answer=source.number in cited,
            )
            for source in context.sources
        ]
        if not cited:
            # 모델이 답을 지어냈거나 표기 지시를 무시한 것 — 근거 사용의 증거가 없다.
            # 생성된 텍스트는 버리지 않고 남긴다. 평가에서 실패 유형을 분류할 재료다.
            return Answer(
                text=text, citations=citations, no_answer_reason=NoAnswerReason.NOT_GROUNDED
            )
        return Answer(text=text, citations=citations)

    def _display_names(self, chunks: list[RetrievedChunk]) -> dict[str, str]:
        names: dict[str, str] = {}
        for result in chunks:
            document_id = result.chunk.document_id
            if document_id in names:
                continue
            document = self._repository.get(document_id)
            if document is not None:
                names[document_id] = document.original_filename
        return names

    @staticmethod
    def _cited_numbers(text: str, context: BuiltContext) -> set[int]:
        """답 속의 [n] 중 실제 근거 번호인 것만 모은다.

        범위 밖 번호는 모델이 지어낸 것이므로 근거 사용으로 치지 않는다.
        """
        valid = {source.number for source in context.sources}
        return {int(match) for match in _CITATION_MARK.findall(text) if int(match) in valid}
