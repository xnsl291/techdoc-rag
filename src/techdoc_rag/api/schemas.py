"""HTTP 요청·응답 모양 (#27).

domain 타입을 그대로 노출하지 않고 여기서 변환한다(DP-43 경계).
domain이 바뀌어도 API 계약은 여기서 의도적으로만 바뀐다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from techdoc_rag.domain.answer import Answer


class ChatRequest(BaseModel):
    # 길이 상한은 라우트에서 settings 값으로 검사한다. Pydantic Field에 박으면
    # 설정 파일이 아니라 코드에 값이 살게 되어 재현성 추적에서 빠진다.
    question: str = Field(min_length=1)


class CitationModel(BaseModel):
    document_id: str
    document_version: int
    display_name: str
    page_start: int
    page_end: int
    chunk_id: str
    is_used_in_answer: bool


class ChatResponse(BaseModel):
    text: str
    citations: list[CitationModel]
    # NOT_GROUNDED일 때 text는 근거 사용이 확인되지 않은 LLM 원문이다.
    # 화면은 answered가 False면 text를 답변처럼 노출하면 안 된다(domain/answer.py).
    no_answer_reason: str | None
    answered: bool

    @classmethod
    def from_answer(cls, answer: Answer) -> ChatResponse:
        return cls(
            text=answer.text,
            citations=[
                CitationModel(
                    document_id=citation.document_id,
                    document_version=citation.document_version,
                    display_name=citation.display_name,
                    page_start=citation.page_start,
                    page_end=citation.page_end,
                    chunk_id=citation.chunk_id,
                    is_used_in_answer=citation.is_used_in_answer,
                )
                for citation in answer.citations
            ],
            no_answer_reason=(
                answer.no_answer_reason.value if answer.no_answer_reason else None
            ),
            answered=answer.is_answered,
        )


class HealthResponse(BaseModel):
    status: str  # "ok" 또는 "degraded"
    components: dict[str, str]  # 구성요소 이름 → "ok" 또는 실패 사유
