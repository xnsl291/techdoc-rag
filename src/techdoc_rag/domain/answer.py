"""답변과 그 근거."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Citation:
    """답변이 참조한 원문 위치.

    이 계약은 첫 구현부터 유지한다. 나중에 끼워 넣으려면 파이프라인 전 구간을 손봐야 한다.

    근거 하나는 세 단계를 거친다. 검색기가 찾고, 근거 예산 안에 들어 프롬프트에
    가고, 답변이 인용한다. **단계마다 떨어져 나간다.** 2026-09-22 실측에서 검색기가
    찾은 평균 24.6개 중 9개만 프롬프트에 갔고, 정답을 찾아 놓고 예산에서 밀린
    문항이 24개 중 4건이었다(#53).

    그래서 검색된 것을 전부 담되 어디까지 갔는지를 두 플래그로 구분한다.

    - reached_prompt: 근거 예산을 통과해 LLM 입력에 들어갔나
    - is_used_in_answer: 답변이 [번호]로 인용했나 (DP-56)

    reached_prompt가 False면 is_used_in_answer는 항상 False다. 프롬프트에 없던
    것을 모델이 인용할 수 없다.

    이 구분이 없으면 "검색이 못 찾은 것"과 "찾았는데 잘린 것"이 같아 보인다.
    고칠 곳이 검색기인지 예산인지 알 수 없게 된다.
    """

    document_id: str
    document_version: int
    display_name: str
    page_start: int
    page_end: int
    chunk_id: str
    is_used_in_answer: bool
    reached_prompt: bool = True


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

    no_answer_reason이 있으면 text는 답변이 아니다. NO_RELEVANT_CHUNK와
    LOW_RELEVANCE의 text는 확인 불가 안내문이고, NOT_GROUNDED의 text는
    근거 사용이 확인되지 않은 LLM 원문이다 — 평가에서 실패 유형을 분류할
    재료로 보존하는 것이며, 표시 계층은 이것을 답변처럼 노출하면 안 된다.
    검색이나 생성이 실패한 경우는 여기로 오지 않고 예외로 처리한다.
    장애를 No-answer로 감추면 품질 문제와 장애를 구분할 수 없게 된다.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    no_answer_reason: NoAnswerReason | None = None

    @property
    def is_answered(self) -> bool:
        return self.no_answer_reason is None
