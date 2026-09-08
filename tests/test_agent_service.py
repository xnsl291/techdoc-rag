"""에이전트 루프 테스트 (#42).

파이프라인에 없던 실패 둘을 여기서 막는다.
- 끝나지 않는 것: 상한에서 끊고 지금까지 모은 근거로 답하게 한다
- 도구를 잘못 부르는 것: 예외로 끊지 않고 알려 준다

인프라 장애는 반대다. 도구 결과로 감추지 않고 그대로 올린다(D-005).
"""

from __future__ import annotations

import json

import pytest

from techdoc_rag.agent.agent_service import NO_ANSWER_TEXT, AgentService
from techdoc_rag.domain.errors import RetrievalError
from techdoc_rag.domain.ports import ChatTurn, ToolCall


class FakeToolBox:
    """정해 둔 결과를 돌려주고 호출을 기록한다."""

    def __init__(self, results: dict[str, str] | None = None, error: Exception | None = None):
        self._results = results or {}
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    def definitions(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "search_manual"}}]

    def call(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        return self._results.get(name, json.dumps({"근거": []}, ensure_ascii=False))


class ScriptedLlm:
    """미리 정한 차례대로 응답한다."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict], max_tokens: int) -> ChatTurn:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._turns:
            return ChatTurn(content="더 할 말 없음", tool_calls=[])
        return self._turns.pop(0)


def _service(llm, toolbox, max_steps: int = 5) -> AgentService:
    return AgentService(
        llm_client=llm, toolbox=toolbox, max_steps=max_steps, max_answer_tokens=256
    )


def test_도구를_안_부르면_그것이_답이다() -> None:
    llm = ScriptedLlm([ChatTurn(content="주위 온도는 -10~50℃입니다 (M100 p.12).", tool_calls=[])])
    toolbox = FakeToolBox()

    answer = _service(llm, toolbox).ask("설치 온도는?")

    assert "-10~50℃" in answer.text
    assert answer.steps == []
    assert toolbox.calls == []


def test_도구를_부르면_결과를_넣고_다시_묻는다() -> None:
    llm = ScriptedLlm(
        [
            ChatTurn(content="", tool_calls=[ToolCall("search_manual", {"question": "정격 전류"})]),
            ChatTurn(content="정격 전류는 0.8~11.0A입니다 (G100 p.248).", tool_calls=[]),
        ]
    )
    toolbox = FakeToolBox({"search_manual": json.dumps({"근거": [{"문서": "g100", "쪽": "248"}]})})

    answer = _service(llm, toolbox).ask("G100 정격 전류는?")

    assert toolbox.calls == [("search_manual", {"question": "정격 전류"})]
    assert answer.used_tools == ["search_manual"]
    assert "0.8~11.0A" in answer.text
    # 두 번째 호출에는 도구 결과가 대화에 들어가 있어야 한다
    second = llm.calls[1]["messages"]
    assert any(m["role"] == "tool" for m in second)


def test_도구를_여러_번_부를_수_있다() -> None:
    llm = ScriptedLlm(
        [
            ChatTurn(content="", tool_calls=[ToolCall("list_documents", {})]),
            ChatTurn(
                content="",
                tool_calls=[
                    ToolCall("compare_spec", {"field": "정격 전류", "documents": ["G100", "M100"]})
                ],
            ),
            ChatTurn(content="두 제품의 정격 전류는 조건이 다릅니다.", tool_calls=[]),
        ]
    )
    toolbox = FakeToolBox()

    answer = _service(llm, toolbox).ask("G100과 M100 비교해줘")

    assert answer.used_tools == ["list_documents", "compare_spec"]
    assert answer.stopped_at_limit is False


def test_한_차례에_도구_둘을_불러도_모두_실행한다() -> None:
    llm = ScriptedLlm(
        [
            ChatTurn(
                content="",
                tool_calls=[
                    ToolCall("search_manual", {"question": "a"}),
                    ToolCall("search_manual", {"question": "b"}),
                ],
            ),
            ChatTurn(content="답", tool_calls=[]),
        ]
    )
    toolbox = FakeToolBox()

    answer = _service(llm, toolbox).ask("질문")

    assert len(toolbox.calls) == 2
    assert len(answer.steps) == 2


def test_끝나지_않으면_상한에서_끊고_모은_근거로_답한다() -> None:
    """모델이 같은 도구를 계속 부르는 것이 에이전트의 기본 실패 모드다.
    조용히 도는 것보다 끊고 알리는 쪽이 낫다."""
    # 상한(3)만큼 도구를 부르고, 마무리 호출에서 답이 나오는 대본.
    # 여기서 더 많이 넣으면 마무리 호출이 그중 하나를 먹어 답이 안 나온다.
    forever = [
        ChatTurn(content="", tool_calls=[ToolCall("search_manual", {"question": "또"})])
        for _ in range(3)
    ]
    llm = ScriptedLlm([*forever, ChatTurn(content="여기까지 찾은 것으로는", tool_calls=[])])
    toolbox = FakeToolBox()

    answer = _service(llm, toolbox, max_steps=3).ask("질문")

    assert answer.stopped_at_limit is True
    assert len(answer.steps) == 3  # 상한만큼만 부른다
    # 마지막 물음에는 도구를 빼서 더 부르지 못하게 한다
    assert llm.calls[-1]["tools"] == []
    assert "여기까지" in answer.text


def test_상한에서_끊겼는데_답도_못_내면_확인_불가다() -> None:
    llm = ScriptedLlm(
        [
            ChatTurn(content="", tool_calls=[ToolCall("search_manual", {"question": "x"})]),
            ChatTurn(content="   ", tool_calls=[]),
        ]
    )

    answer = _service(llm, FakeToolBox(), max_steps=1).ask("질문")

    assert answer.text == NO_ANSWER_TEXT
    assert answer.stopped_at_limit is True


def test_빈_답변은_확인_불가로_바꾼다() -> None:
    llm = ScriptedLlm([ChatTurn(content="", tool_calls=[])])

    answer = _service(llm, FakeToolBox()).ask("질문")

    assert answer.text == NO_ANSWER_TEXT


def test_인프라_장애는_도구_결과로_감추지_않는다() -> None:
    """검색 저장소에 못 붙는 것을 결과로 삼키면 모델이 '근거가 없다'고 답한다.
    품질 문제와 장애를 구분할 수 없게 된다(D-005)."""
    llm = ScriptedLlm(
        [ChatTurn(content="", tool_calls=[ToolCall("search_manual", {"question": "x"})])]
    )
    toolbox = FakeToolBox(error=RetrievalError("Qdrant 접근 불가"))

    with pytest.raises(RetrievalError):
        _service(llm, toolbox).ask("질문")


def test_도구_호출과_결과를_모두_남긴다() -> None:
    """남기지 않으면 왜 그런 답이 나왔는지 되짚을 수 없다."""
    llm = ScriptedLlm(
        [
            ChatTurn(content="", tool_calls=[ToolCall("search_manual", {"question": "정격"})]),
            ChatTurn(content="답", tool_calls=[]),
        ]
    )
    toolbox = FakeToolBox({"search_manual": json.dumps({"근거": [{"문서": "g100", "쪽": "248"}]})})

    answer = _service(llm, toolbox).ask("질문")

    step = answer.steps[0]
    assert step.tool == "search_manual"
    assert step.arguments == {"question": "정격"}
    assert "g100" in step.result


def test_도구_결과에서_근거_쪽을_모은다() -> None:
    llm = ScriptedLlm(
        [
            ChatTurn(content="", tool_calls=[ToolCall("compare_spec", {})]),
            ChatTurn(content="답", tool_calls=[]),
        ]
    )
    toolbox = FakeToolBox(
        {
            "compare_spec": json.dumps(
                {
                    "문서별 근거": {
                        "G100": [{"문서": "ls-g100-v1", "쪽": "248"}],
                        "M100": [{"문서": "ls-m100-v1", "쪽": "172"}],
                    }
                },
                ensure_ascii=False,
            )
        }
    )

    answer = _service(llm, toolbox).ask("비교")

    assert AgentService.evidence_pages(answer.steps) == [
        "ls-g100-v1 p.248",
        "ls-m100-v1 p.172",
    ]
