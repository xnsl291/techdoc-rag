"""거절 표현이 섞인 답을 세는 부분 테스트 (#50).

`scripts/evaluate.py`는 패키지가 아니라 실행 스크립트라서 경로로 불러온다.
스크립트 전체를 시험하는 것이 아니라 `hedged` 하나만 본다. 이 함수는
**자동으로 거절 판정을 내리지 않는다**는 것이 요점이고, 누가 나중에 "그냥
거절로 뒤집으면 되지 않나" 하고 고치면 오거절 수치가 조용히 부풀려진다.

아래 네 건은 2026-09-22 평가 실행의 실제 답변 원문이다. 지어낸 것이 아니다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
assert _SPEC is not None and _SPEC.loader is not None
evaluate = importlib.util.module_from_spec(_SPEC)
# exec_module 전에 등록해야 한다. dataclasses가 타입 힌트를 풀 때
# sys.modules[cls.__module__]을 찾는데, 등록이 없으면 None이 돌아와 터진다.
sys.modules[_SPEC.name] = evaluate
_SPEC.loader.exec_module(evaluate)


def _record(text: str, answered: bool = True) -> dict:
    return {"answered": answered, "text": text}


# 실제 답변 원문에서 앞부분만 옮겼다. 네 건 모두 같은 형태로 시작한다.
TRUE_ANSWER_Q020 = (
    "제공된 근거 자료에는 S100 인버터의 Over Current1 트립이 정격 전류 대비 몇 "
    "퍼센트에서 발생하는지에 대한 구체적인 수치나 조건은 명시되어 있지 않습니다. "
    "근거 [7] 에서는 \"인버터 출력 전류가 정격 전류의 200% 이상일 때 발생\""
)
TRUE_ANSWER_Q014 = (
    "제공된 근거 자료에는 H100 인버터의 냉각 팬 교체 기준 시간에 대한 구체적인 "
    "값이나 계산식이 명시되어 있지 않습니다. 근거 [1]과 [3], [5] 에 따르면 "
    "'Fan 수명 진단' 기능은 다음과 같이 작동합니다"
)
REAL_REFUSAL_Q018 = (
    "제공된 근거 자료에는 S100 인버터의 'IO Board Trip'가 몇 초 이상 지속되어 "
    "발생하는지에 대한 구체적인 시간 조건이나 설정 값이 포함되어 있지 않습니다. "
    "따라서 이 질문은 근거 자료에서 확인할 수 없습니다."
)
# 이 줄은 원문을 끝까지 옮긴다. 앞부분만 잘라 넣었더니 맨 뒤의 "근거 자료에서
# 확인할 수 없습니다"가 빠져 표현이 하나도 안 걸렸다. 실제 데이터를 잘라 쓰면
# 잘린 것이 원문과 다르게 동작한다.
REAL_REFUSAL_Q025 = (
    "제공된 근거 자료에는 iS7 인버터의 제동 저항 사용 시 **연속적으로 사용할 수 있는 "
    "최대 시간**에 대한 명시적인 정보가 없습니다. [2], [5], [8] 번 근거에서는 제동 "
    "저항기의 와트값, 제동 토크, 외형 치수 등 관련 사양은 제시되어 있으나, 연속 운전 "
    "시간을 언급하지는 않았습니다. 따라서 근거 자료에서 확인할 수 없습니다."
)


@pytest.mark.parametrize(
    "text",
    [TRUE_ANSWER_Q020, TRUE_ANSWER_Q014, REAL_REFUSAL_Q018, REAL_REFUSAL_Q025],
)
def test_네_건_모두_섞인_것으로_표시된다(text: str) -> None:
    """앞의 둘은 진짜 답변이고 뒤의 둘은 진짜 거절인데 문장 형태가 같다.
    문자열로는 못 가르므로 넷 다 "사람이 봐야 함"으로 표시한다."""
    assert evaluate.hedged(_record(text)) is True


# 표현 목록의 항목마다 그것 하나만 걸리는 실제 답을 골라 두었다. 저장된 실행
# 결과를 전부 훑어 뽑은 것이다. 항목을 지우면 그 줄이 죽는다.
ONLY_확인할수없 = (
    "제공된 근거 자료에는 G100 인버터의 파라미터 초기화 시 트립 발생 여부에 대한 "
    "명시적인 설명이 없습니다. 근거 자료에서 확인할 수 없습니다"
)
ONLY_포함되어있지않 = (
    "제공해주신 문서에는 LSLV0075M100 모델의 가격 정보가 포함되어 있지 않습니다."
)
ONLY_명시되어있지않 = (
    "제공된 근거에 따르면 G100 인버터의 파라미터 초기화 기능은 트립 발생 "
    "상태에서도 수행할 수 있습니다. 다만 조건은 명시되어 있지 않습니다"
)
ONLY_나와있지않 = (
    "제공해주신 문서만으로는 지멘스 인버터의 구체적인 사양 데이터가 명시적으로 "
    "나와 있지 않아, 두 제품 간의 효율을 직접 비교할 수 없"
)


@pytest.mark.parametrize(
    "text",
    [ONLY_확인할수없, ONLY_포함되어있지않, ONLY_명시되어있지않, ONLY_나와있지않],
)
def test_표현_목록의_항목마다_단독으로_걸린다(text: str) -> None:
    """항목 하나를 지우면 이 중 한 줄이 죽어야 한다. 그래야 목록이 시험된다."""
    assert evaluate.hedged(_record(text)) is True


def test_평범한_답변은_표시되지_않는다() -> None:
    plain = "H100 RTC용 배터리 수명은 약 3년입니다 [2]."

    assert evaluate.hedged(_record(plain)) is False


def test_이미_거절로_분류된_답은_세지_않는다() -> None:
    """이 수치는 "답한 것으로 분류됐는데 의심스러운 건"을 세는 것이다.
    이미 거절로 잡힌 것까지 더하면 거절을 두 번 세게 된다."""
    assert evaluate.hedged(_record(REAL_REFUSAL_Q018, answered=False)) is False


def test_집계가_섞인_건수를_함께_낸다() -> None:
    records = [
        {"id": "a", "answerable": True, "answered": True, "text": TRUE_ANSWER_Q020,
         "retrieval_hit": True, "prompt_hit": True, "citation_hit": True, "seconds": 1.0},
        {"id": "b", "answerable": True, "answered": True, "text": "정격은 5A입니다 [1].",
         "retrieval_hit": True, "prompt_hit": True, "citation_hit": True, "seconds": 1.0},
        {"id": "c", "answerable": False, "answered": False, "text": REAL_REFUSAL_Q018,
         "retrieval_hit": False, "prompt_hit": False, "citation_hit": False, "seconds": 1.0},
    ]

    summary = evaluate.score(records)

    assert summary["거절문구섞임"] == 1
