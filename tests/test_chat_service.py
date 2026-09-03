"""ChatService 테스트 (#19).

LLM·검색은 가짜다. 여기서 검증하는 것은 No-answer 세 갈래 분기,
인용 번호 → is_used_in_answer 매핑, 장애를 No-answer로 감추지 않는 것.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from techdoc_rag.domain.answer import NoAnswerReason
from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import GenerationError, RetrievalError
from techdoc_rag.query.chat_service import ChatService
from techdoc_rag.query.context_builder import ContextBuilder
from techdoc_rag.query.retriever import RetrievalResult


def _retrieved(chunk_id: str, score: float, text: str = "정격 전류는 5A다") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="ls-m100-v1",
            document_version=1,
            page_start=42,
            page_end=42,
            text=text,
        ),
        score=score,
    )


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.fail = False

    def retrieve(self, question: str) -> RetrievalResult:
        if self.fail:
            raise RetrievalError("저장소 접근 불가")
        return self._result


class FakeLlm:
    model_name = "fake-llm"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []
        self.fail = False

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]:
        if self.fail:
            raise GenerationError("LLM 서버 다운")
        self.prompts.append(prompt)
        # 스트리밍처럼 두 조각으로 나눠 낸다.
        half = len(self._reply) // 2
        yield self._reply[:half]
        yield self._reply[half:]


class FakeRepository:
    def get(self, document_id: str) -> Document | None:
        if document_id != "ls-m100-v1":
            return None
        return Document(
            document_id="ls-m100-v1",
            logical_document_id="ls-m100",
            document_version=1,
            original_filename="M100 사용설명서.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            file_size_bytes=1,
            page_count=270,
            document_type="manual",
            status=DocumentStatus.READY,
            is_active=True,
            created_at=datetime.now(UTC),
        )


def _service(retriever: FakeRetriever, llm: FakeLlm) -> ChatService:
    return ChatService(
        retriever=retriever,
        context_builder=ContextBuilder(budget_chars=5000),
        llm_client=llm,
        repository=FakeRepository(),
        max_answer_tokens=256,
    )


def _found(*chunks: RetrievedChunk, dropped: int = 0) -> RetrievalResult:
    return RetrievalResult(chunks=list(chunks), dropped_below_threshold=dropped)


def test_답과_인용이_나오고_사용된_근거만_표시된다() -> None:
    llm = FakeLlm("정격 전류는 5A입니다 [1].")
    service = _service(
        FakeRetriever(_found(_retrieved("a", 0.9), _retrieved("b", 0.7, "무관한 내용"))), llm
    )

    answer = service.ask("정격 전류는?")

    assert answer.is_answered
    assert "5A" in answer.text
    used = {c.chunk_id: c.is_used_in_answer for c in answer.citations}
    assert used == {"a": True, "b": False}  # 가져온 것 전부가 아니라 [1]만 사용 표시
    citation = answer.citations[0]
    assert citation.display_name == "M100 사용설명서.pdf"
    assert (citation.page_start, citation.page_end) == (42, 42)


def test_프롬프트에_근거와_질문이_들어간다() -> None:
    llm = FakeLlm("답 [1].")
    service = _service(FakeRetriever(_found(_retrieved("a", 0.9))), llm)

    service.ask("정격 전류는?")

    prompt = llm.prompts[0]
    assert "정격 전류는 5A다" in prompt  # 근거 본문
    assert "정격 전류는?" in prompt  # 질문
    assert "근거에 없는 내용은 답하지 마라" in prompt


def test_검색_0건은_NO_RELEVANT_CHUNK다() -> None:
    llm = FakeLlm("호출되면 안 됨")
    service = _service(FakeRetriever(_found()), llm)

    answer = service.ask("질문")

    assert answer.no_answer_reason is NoAnswerReason.NO_RELEVANT_CHUNK
    assert llm.prompts == []  # 근거 없이 LLM을 부르지 않는다


def test_전부_임계값_미만이면_LOW_RELEVANCE다() -> None:
    service = _service(FakeRetriever(_found(dropped=3)), FakeLlm("호출되면 안 됨"))

    answer = service.ask("질문")

    assert answer.no_answer_reason is NoAnswerReason.LOW_RELEVANCE


def test_인용_번호가_없으면_NOT_GROUNDED다() -> None:
    """표기 지시를 무시한 답은 근거 사용의 증거가 없다. 텍스트는 평가 재료로 남긴다."""
    llm = FakeLlm("이 인버터의 정격 전류는 아마 10A일 것입니다.")
    service = _service(FakeRetriever(_found(_retrieved("a", 0.9))), llm)

    answer = service.ask("정격 전류는?")

    assert answer.no_answer_reason is NoAnswerReason.NOT_GROUNDED
    assert "10A" in answer.text
    assert all(not c.is_used_in_answer for c in answer.citations)


def test_범위_밖_번호는_인용으로_치지_않는다() -> None:
    llm = FakeLlm("정격 전류는 5A입니다 [7].")  # 근거는 1개뿐인데 [7]
    service = _service(FakeRetriever(_found(_retrieved("a", 0.9))), llm)

    answer = service.ask("정격 전류는?")

    assert answer.no_answer_reason is NoAnswerReason.NOT_GROUNDED


def test_검색_실패는_예외로_올라간다() -> None:
    """장애를 No-answer로 감추면 품질 문제와 장애를 구분할 수 없다(D-005)."""
    retriever = FakeRetriever(_found())
    retriever.fail = True
    service = _service(retriever, FakeLlm("호출되면 안 됨"))

    with pytest.raises(RetrievalError):
        service.ask("질문")


def test_생성_실패는_예외로_올라간다() -> None:
    llm = FakeLlm("무관")
    llm.fail = True
    service = _service(FakeRetriever(_found(_retrieved("a", 0.9))), llm)

    with pytest.raises(GenerationError):
        service.ask("질문")
