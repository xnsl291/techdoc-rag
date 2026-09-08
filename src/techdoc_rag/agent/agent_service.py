"""도구를 골라 여러 번 부르는 에이전트 (#42).

기존 질의 서비스는 순서가 고정돼 있다 — 검색 한 번, 근거 조립, 답 하나.
그 구조로는 "G100과 M100의 정격 전류를 비교해줘"를 못 한다. 몇 번을 어떤
순서로 찾을지가 질문마다 다르기 때문이다. 여기서는 순서를 모델이 정한다.

**파이프라인에 없던 실패가 둘 생긴다.** 계약으로 막는다.

1. **끝나지 않는다.** 모델이 같은 도구를 계속 부르거나 답을 못 낸다.
   `max_steps`에서 끊고, 그때까지 모은 근거로 답하게 한다. 그것도 못 하면
   확인 불가로 끝낸다. 조용히 도는 것보다 끊고 알리는 쪽이 낫다.
2. **도구를 잘못 부른다.** 없는 도구, 빈 인자, 없는 문서. 예외로 끊지 않고
   결과에 적어 돌려준다 — 모델이 다음 차례에 고쳐 부를 수 있다.
   다만 **인프라 장애는 그대로 올린다**(D-005). 검색 저장소에 못 붙는 것을
   도구 결과로 감추면 모델이 "근거가 없다"고 답한다.

**모든 도구 호출과 결과를 남긴다.** 남기지 않으면 왜 그런 답이 나왔는지
되짚을 수 없다. 화면에서 이 기록을 보여 주는 것이 이 기능의 핵심이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from techdoc_rag.agent.tools import ToolBox
from techdoc_rag.domain.ports import ToolCallingClient

SYSTEM_PROMPT = """너는 제품 매뉴얼을 찾아 답하는 도우미다.

지킬 것:
1. 답은 반드시 도구로 찾은 근거에서만 만든다. 근거에 없으면 지어내지 마라.
2. 어떤 제품이 있는지 모르면 list_documents로 먼저 확인한다.
3. 둘 이상의 제품을 비교할 때는 compare_spec을 쓴다. 전체 검색은 한 제품에
   치우쳐서 비교가 성립하지 않는다.
4. 값이 다르다고 바로 충돌이라고 하지 마라. 조건(모델·용량·측정 기준)과
   단위가 같은지 먼저 확인하고, 조건이 다르면 조건과 함께 제시한다.
5. 답에는 어느 문서 몇 쪽에서 나온 값인지 함께 적는다.
6. 근거를 찾지 못하면 "근거 자료에서 확인할 수 없습니다"라고 답한다."""

# 끊기 직전에 넣는 지시. 도구를 더 부르지 못하게 하고 지금까지 모은 것으로 답하게 한다.
_WRAP_UP = (
    "도구를 더 부르지 말고, 지금까지 찾은 근거만으로 답하라. "
    "근거가 모자라면 무엇이 부족한지 밝히고 확인할 수 없다고 답하라."
)

NO_ANSWER_TEXT = "근거 자료에서 확인할 수 없습니다."


@dataclass(frozen=True, slots=True)
class AgentStep:
    """도구 한 번 호출과 그 결과."""

    tool: str
    arguments: dict
    result: str


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    text: str
    steps: list[AgentStep] = field(default_factory=list)
    # 상한에 걸려 끊겼는지. 화면과 평가에서 "덜 찾은 답"을 구분하는 데 쓴다.
    stopped_at_limit: bool = False

    @property
    def used_tools(self) -> list[str]:
        return [step.tool for step in self.steps]


class AgentService:
    def __init__(
        self,
        llm_client: ToolCallingClient,
        toolbox: ToolBox,
        max_steps: int,
        max_answer_tokens: int,
    ) -> None:
        self._llm_client = llm_client
        self._toolbox = toolbox
        self._max_steps = max_steps
        self._max_answer_tokens = max_answer_tokens

    def ask(self, question: str) -> AgentAnswer:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools = self._toolbox.definitions()
        steps: list[AgentStep] = []

        for remaining in range(self._max_steps, 0, -1):
            turn = self._llm_client.chat(
                messages=messages, tools=tools, max_tokens=self._max_answer_tokens
            )
            if not turn.tool_calls:
                # 도구를 안 부르면 답을 낸 것이다. 내용이 비어 있으면 모델이
                # 아무 말도 못 한 것이므로 확인 불가로 끝낸다.
                return AgentAnswer(text=turn.content.strip() or NO_ANSWER_TEXT, steps=steps)

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                        for call in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                result = self._toolbox.call(call.name, call.arguments)
                steps.append(
                    AgentStep(tool=call.name, arguments=call.arguments, result=result)
                )
                messages.append({"role": "tool", "content": result, "tool_name": call.name})

            if remaining == 1:
                break

        # 상한에 걸렸다. 도구 없이 한 번 더 물어 지금까지 모은 근거로 답하게 한다.
        messages.append({"role": "user", "content": _WRAP_UP})
        final = self._llm_client.chat(
            messages=messages, tools=[], max_tokens=self._max_answer_tokens
        )
        return AgentAnswer(
            text=final.content.strip() or NO_ANSWER_TEXT, steps=steps, stopped_at_limit=True
        )

    @staticmethod
    def evidence_pages(steps: list[AgentStep]) -> list[str]:
        """도구가 돌려준 근거의 문서와 쪽을 모은다.

        청크 단위 Citation과는 다르다. 도구 결과에는 청크 ID가 없고 문서와
        쪽만 있다. 답변에 실제로 쓰였는지도 여기서는 알 수 없다 — 그 판정은
        질의 서비스의 인용 번호 방식(DP-56)이 하는 일이고, 에이전트 경로에
        같은 장치를 붙이는 것은 아직 안 했다.
        """
        pages: list[str] = []
        for step in steps:
            try:
                payload = json.loads(step.result)
            except json.JSONDecodeError:
                continue
            groups = [payload.get("근거") or []]
            groups.extend((payload.get("문서별 근거") or {}).values())
            for group in groups:
                for item in group:
                    label = f"{item.get('문서')} p.{item.get('쪽')}"
                    if label not in pages:
                        pages.append(label)
        return pages
