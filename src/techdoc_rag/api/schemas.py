"""HTTP 요청·응답 모양 (#27).

domain 타입을 그대로 노출하지 않고 여기서 변환한다(DP-43 경계).
domain이 바뀌어도 API 계약은 여기서 의도적으로만 바뀐다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from techdoc_rag.agent.agent_service import AgentAnswer, AgentService
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


# 도구 결과 하나를 응답에 담을 때의 길이 상한. compare_spec이 돌려주는 근거 수는
# 모델이 정한 문서 수만큼 늘어나므로, 상한이 없으면 응답 크기를 예측할 수 없다.
#
# 2026-09-08 실측(문서 2권, top_k=8): list_documents 134자,
# compare_spec 2문서 3,744~3,778자, search_manual 전체 11,267자.
# 처음 정한 2,000자는 평범한 2문서 비교에서 이미 걸려서, 병리적인 경우가 아니라
# 정상 경우를 자르고 있었다. 실측 최대값이 들어가고 문서를 많이 지정한
# compare_spec은 여전히 막히도록 12,000자로 잡는다.
MAX_STEP_RESULT_CHARS = 12000
TRUNCATION_MARK = "…(이하 잘림)"


class AgentStepModel(BaseModel):
    """도구 한 번 호출과 그 결과.

    result는 감사와 화면 표시용이다. 상한에서 자르므로 JSON으로 다시 파싱할 수
    있다고 보장하지 않는다. 값이 필요하면 evidence_pages를 본다.
    """

    tool: str
    arguments: dict
    result: str


class AgentResponse(BaseModel):
    """에이전트 답변.

    ChatResponse를 재사용하지 않는다. 그쪽 citations의 is_used_in_answer는
    본문의 인용 번호로 확인한 값인데(DP-56), 에이전트 경로에는 그 장치가 없다.
    같은 필드에 담으면 확인하지 않은 것을 확인한 것처럼 내보내게 된다.
    answered·no_answer_reason도 에이전트가 만들지 않는 값이다.
    """

    text: str
    steps: list[AgentStepModel]
    # 도구 호출 상한에서 끊겼는지. 화면에서 "덜 찾은 답"을 구분하는 데 쓴다.
    stopped_at_limit: bool
    # 도구 결과에서 모은 "문서 p.쪽". 답변에 실제로 쓰였는지까지는 알 수 없다.
    evidence_pages: list[str]

    @classmethod
    def from_answer(cls, answer: AgentAnswer) -> AgentResponse:
        return cls(
            text=answer.text,
            steps=[
                AgentStepModel(
                    tool=step.tool, arguments=step.arguments, result=_clip(step.result)
                )
                for step in answer.steps
            ],
            stopped_at_limit=answer.stopped_at_limit,
            evidence_pages=AgentService.evidence_pages(answer.steps),
        )


def _clip(result: str) -> str:
    if len(result) <= MAX_STEP_RESULT_CHARS:
        return result
    return result[:MAX_STEP_RESULT_CHARS] + TRUNCATION_MARK


class HealthResponse(BaseModel):
    status: str  # "ok" 또는 "degraded"
    components: dict[str, str]  # 구성요소 이름 → "ok" 또는 실패 사유
