"""FastAPI 경계 테스트 (#27).

chat_service는 가짜다 — 파이프라인 내부는 각자의 테스트가 덮고,
여기서 보는 것은 HTTP 계약: 응답 모양, 오류 매핑(장애=503, 검증=422),
No-answer가 오류가 아니라 200인 것.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from techdoc_rag.api.app import create_app
from techdoc_rag.domain.answer import Answer, Citation, NoAnswerReason
from techdoc_rag.domain.errors import GenerationError, RetrievalError


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


def _client(
    service: FakeChatService,
    probes: dict | None = None,
    max_question_chars: int = 100,
) -> TestClient:
    app = create_app(
        chat_service=service,
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


@pytest.mark.parametrize("error", [RetrievalError("Qdrant 접근 불가"), GenerationError("LLM 다운")])
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
