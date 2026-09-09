"""FastAPI 경계 테스트 (#27).

chat_service는 가짜다 — 파이프라인 내부는 각자의 테스트가 덮고,
여기서 보는 것은 HTTP 계약: 응답 모양, 오류 매핑(장애=503, 검증=422),
No-answer가 오류가 아니라 200인 것.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from techdoc_rag.agent.agent_service import AgentAnswer, AgentStep
from techdoc_rag.api.app import create_app
from techdoc_rag.api.schemas import MAX_STEP_RESULT_CHARS, TRUNCATION_MARK
from techdoc_rag.domain.answer import Answer, Citation, NoAnswerReason
from techdoc_rag.domain.errors import (
    GenerationError,
    MetadataStoreError,
    RetrievalError,
)


class FakeChatService:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.questions: list[str] = []

    def ask(self, question: str) -> Answer:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        assert self._answer is not None
        return self._answer


def _answered() -> Answer:
    return Answer(
        text="정격 전류는 5A입니다 [1].",
        citations=[
            Citation(
                document_id="ls-m100-v1",
                document_version=1,
                display_name="M100 사용설명서.pdf",
                page_start=42,
                page_end=43,
                chunk_id="ls-m100-v1:0007",
                is_used_in_answer=True,
            )
        ],
    )


class FakeAgentService:
    def __init__(self, answer: AgentAnswer | None = None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.questions: list[str] = []

    def ask(self, question: str) -> AgentAnswer:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return self._answer if self._answer is not None else AgentAnswer(text="답")


def _client(
    service: FakeChatService,
    probes: dict | None = None,
    max_question_chars: int = 100,
    agent_service: FakeAgentService | None = None,
) -> TestClient:
    app = create_app(
        chat_service=service,
        agent_service=agent_service if agent_service is not None else FakeAgentService(),
        health_probes=probes if probes is not None else {"sqlite": lambda: None},
        max_question_chars=max_question_chars,
    )
    # 서버 오류를 예외로 터뜨리지 않고 상태 코드로 받는다 — 그게 검증 대상이다.
    return TestClient(app, raise_server_exceptions=False)


def test_답변이_JSON_계약대로_나온다() -> None:
    client = _client(FakeChatService(answer=_answered()))

    response = client.post("/chat", json={"question": "정격 전류는?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["no_answer_reason"] is None
    assert "5A" in body["text"]
    assert body["citations"] == [
        {
            "document_id": "ls-m100-v1",
            "document_version": 1,
            "display_name": "M100 사용설명서.pdf",
            "page_start": 42,
            "page_end": 43,
            "chunk_id": "ls-m100-v1:0007",
            "is_used_in_answer": True,
        }
    ]


def test_No_answer는_오류가_아니라_200이다() -> None:
    service = FakeChatService(
        answer=Answer(text="원문", no_answer_reason=NoAnswerReason.NOT_GROUNDED)
    )
    client = _client(service)

    response = client.post("/chat", json={"question": "질문"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["no_answer_reason"] == "NOT_GROUNDED"


@pytest.mark.parametrize(
    "error",
    [
        RetrievalError("Qdrant 접근 불가"),
        GenerationError("LLM 다운"),
        MetadataStoreError("SQLite 잠김"),
    ],
)
def test_장애는_503이다(error: Exception) -> None:
    """장애가 No-answer(200)로 둔갑하면 감시가 품질 문제와 장애를 구분 못 한다(D-005)."""
    client = _client(FakeChatService(error=error))

    response = client.post("/chat", json={"question": "질문"})

    assert response.status_code == 503


def test_빈_질문과_긴_질문은_422다() -> None:
    service = FakeChatService(answer=_answered())
    client = _client(service, max_question_chars=10)

    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "가" * 11}).status_code == 422
    assert service.questions == []  # 검증 실패 시 서비스까지 가지 않는다


def test_health_전부_정상이면_ok() -> None:
    client = _client(
        FakeChatService(answer=_answered()),
        probes={"sqlite": lambda: None, "qdrant": lambda: None},
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "components": {"sqlite": "ok", "qdrant": "ok"}}


def test_health_하나라도_죽으면_503이고_어느_것인지_알려준다() -> None:
    def broken() -> None:
        raise ConnectionError("연결 거부")

    client = _client(
        FakeChatService(answer=_answered()),
        probes={"sqlite": lambda: None, "ollama": broken},
    )

    response = client.get("/health")

    assert response.status_code == 503
    components = response.json()["detail"]["components"]
    assert components["sqlite"] == "ok"
    assert components["ollama"].startswith("실패:")


# --- POST /agent (#42) ---
# /chat과 나눠 둔 경로다. 여기서 보는 것은 도구 호출 기록이 응답에 실려
# 나가는가, 그리고 오류 매핑이 /chat과 같은가이다.


def _agent_answer() -> AgentAnswer:
    return AgentAnswer(
        text="G100은 -10~50℃, M100은 -10~50℃입니다.",
        steps=[
            AgentStep(tool="list_documents", arguments={}, result='{"문서": []}'),
            AgentStep(
                tool="compare_spec",
                arguments={"field": "주위 온도", "documents": ["G100", "M100"]},
                result='{"문서별 근거": {"G100": [{"문서": "ls-g100-v1", "쪽": "17~26"}]}}',
            ),
        ],
    )


def test_에이전트_응답에_도구_호출_기록이_들어간다() -> None:
    """이 기록이 화면에 나오는 것이 #42의 목적이다. 응답에서 빠지면
    무엇을 근거로 답했는지 사용자가 확인할 방법이 없다."""
    client = _client(FakeChatService(), agent_service=FakeAgentService(_agent_answer()))

    response = client.post("/agent", json={"question": "G100과 M100 주위 온도 비교"})

    assert response.status_code == 200
    body = response.json()
    assert [step["tool"] for step in body["steps"]] == ["list_documents", "compare_spec"]
    assert body["steps"][1]["arguments"]["documents"] == ["G100", "M100"]
    assert body["evidence_pages"] == ["ls-g100-v1 p.17~26"]
    assert body["stopped_at_limit"] is False


def test_상한에_걸린_답은_그_사실을_함께_보낸다() -> None:
    """덜 찾고 끊긴 답과 다 찾은 답을 화면에서 구분해야 한다."""
    answer = AgentAnswer(text="여기까지", steps=[], stopped_at_limit=True)
    client = _client(FakeChatService(), agent_service=FakeAgentService(answer))

    body = client.post("/agent", json={"question": "질문"}).json()

    assert body["stopped_at_limit"] is True


def test_긴_도구_결과는_잘라서_보낸다() -> None:
    """도구가 돌려주는 근거 수는 모델이 정한 문서 수만큼 늘어난다.
    상한이 없으면 응답 크기를 예측할 수 없다."""
    long_result = "가" * (MAX_STEP_RESULT_CHARS * 2)
    answer = AgentAnswer(
        text="답",
        steps=[AgentStep(tool="search_manual", arguments={}, result=long_result)],
    )
    client = _client(FakeChatService(), agent_service=FakeAgentService(answer))

    sent = client.post("/agent", json={"question": "질문"}).json()["steps"][0]["result"]

    assert len(sent) == MAX_STEP_RESULT_CHARS + len(TRUNCATION_MARK)
    assert sent.endswith(TRUNCATION_MARK)


def test_에이전트_경로도_장애는_503이다() -> None:
    """도구 오용은 에이전트가 결과로 알려 주고 넘어가지만, 검색 저장소에
    못 붙는 것은 그대로 올라와야 한다(D-005)."""
    client = _client(
        FakeChatService(),
        agent_service=FakeAgentService(error=RetrievalError("Qdrant 접근 불가")),
    )

    response = client.post("/agent", json={"question": "질문"})

    assert response.status_code == 503
    assert "Qdrant" in response.json()["detail"]


def test_에이전트_경로도_질문_길이_상한을_적용한다() -> None:
    agent = FakeAgentService()
    client = _client(FakeChatService(), max_question_chars=10, agent_service=agent)

    response = client.post("/agent", json={"question": "가" * 11})

    assert response.status_code == 422
    assert agent.questions == []  # 상한을 넘으면 에이전트를 부르지 않는다
