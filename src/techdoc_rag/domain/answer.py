"""답변과 그 근거."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Citation:
    """답변이 참조한 원문 위치.

    이 계약은 첫 구현부터 유지한다. 나중에 끼워 넣으려면 파이프라인 전 구간을 손봐야 한다.

    is_used_in_answer는 검색된 근거와 실제로 답변에 반영된 근거를 구분한다.
    Top-K로 가져온 것을 전부 근거로 표시하면 Traceability가 의미를 잃는다.
    """

    document_id: str
    document_version: int
    display_name: str
    page_start: int
    page_end: int
    chunk_id: str
    is_used_in_answer: bool


class NoAnswerReason(StrEnum):
    """등록 문서에서 답을 확인할 수 없는 이유.

    이유를 구분해 두면 평가에서 실패 유형을 분류할 수 있다.
    검색이 아무것도 못 찾은 것과, 찾았으나 근거가 약한 것은 개선 방향이 다르다.
    """

    NO_RELEVANT_CHUNK = "NO_RELEVANT_CHUNK"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    NOT_GROUNDED = "NOT_GROUNDED"


@dataclass(frozen=True, slots=True)
class Answer:
    """사용자 질문에 대한 최종 응답.

    no_answer_reason이 있으면 text는 답변이 아니라 확인 불가 안내다.
    검색이나 생성이 실패한 경우는 여기로 오지 않고 예외로 처리한다.
    장애를 No-answer로 감추면 품질 문제와 장애를 구분할 수 없게 된다.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    no_answer_reason: NoAnswerReason | None = None

    @property
    def is_answered(self) -> bool:
        return self.no_answer_reason is None
