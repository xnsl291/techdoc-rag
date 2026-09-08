"""에이전트 도구 테스트 (#42).

여기서 지키는 것 셋.
- 이름 해석: 모델은 "G100"이라 부르고 저장소는 "ls-g100-v1"로 갖고 있다
- 비교는 문서마다 따로 검색한다. 전체 검색 한 번으로는 한 문서에 치우친다
- 도구가 판정하지 않는다. 근거만 주고 같은지 다른지는 모델이 말한다(DP-39)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from techdoc_rag.agent.tools import ToolBox
from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.query.retriever import RetrievalResult


def _document(document_id: str, filename: str) -> Document:
    return Document(
        document_id=document_id,
        logical_document_id=document_id.rsplit("-v", 1)[0],
        document_version=1,
        original_filename=filename,
        sha256="a" * 64,
        mime_type="application/pdf",
        file_size_bytes=1,
        page_count=300,
        document_type="manual",
        status=DocumentStatus.READY,
        is_active=True,
        created_at=datetime.now(UTC),
    )


class FakeRepository:
    def __init__(self) -> None:
        self._documents = {
            "ls-m100-v1": _document("ls-m100-v1", "M100_사용설명서.pdf"),
            "ls-g100-v1": _document("ls-g100-v1", "G100(C)_사용설명서.pdf"),
        }

    def active_document_ids(self) -> list[str]:
        return list(self._documents)

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)


class FakeRetriever:
    """문서별로 다른 결과를 돌려주고, 어떤 범위로 불렸는지 기록한다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> RetrievalResult:
        self.calls.append((question, document_ids))
        target = (document_ids or ["ls-g100-v1"])[0]
        chunk = Chunk(
            chunk_id=f"{target}:0001",
            document_id=target,
            document_version=1,
            page_start=248,
            page_end=250,
            text=f"{target}의 정격 전류 표",
        )
        return RetrievalResult(
            chunks=[RetrievedChunk(chunk=chunk, score=0.9)], dropped_below_threshold=0
        )


def _toolbox() -> tuple[ToolBox, FakeRetriever]:
    retriever = FakeRetriever()
    return ToolBox(retriever=retriever, repository=FakeRepository()), retriever


def test_도구_정의에_필요한_셋이_있다() -> None:
    box, _ = _toolbox()

    names = [d["function"]["name"] for d in box.definitions()]

    assert names == ["list_documents", "search_manual", "compare_spec"]


def test_문서_목록을_돌려준다() -> None:
    box, _ = _toolbox()

    result = json.loads(box.call("list_documents", {}))

    assert {d["id"] for d in result["문서"]} == {"ls-m100-v1", "ls-g100-v1"}


def test_제품명으로_문서를_찾는다() -> None:
    """모델은 'G100'이라 부르고 저장소는 'ls-g100-v1'로 갖고 있다."""
    box, retriever = _toolbox()

    box.call("search_manual", {"question": "정격 전류", "document": "G100"})

    assert retriever.calls == [("정격 전류", ["ls-g100-v1"])]


def test_없는_문서는_오류로_알려_주고_힌트를_준다() -> None:
    """예외로 끊으면 에이전트 루프가 죽는다. 알려 주면 다음 차례에 고쳐 부른다."""
    box, retriever = _toolbox()

    result = json.loads(box.call("search_manual", {"question": "정격", "document": "S999"}))

    assert "그런 문서가 없음" in result["오류"]
    assert "list_documents" in result["힌트"]
    assert retriever.calls == []  # 검색까지 가지 않는다


def test_비교는_문서마다_따로_검색한다() -> None:
    """전체 검색 한 번으로는 점수 높은 한 문서가 상위를 차지한다
    (2026-09-08 실측: '정격 전류' 상위 6개가 전부 G100)."""
    box, retriever = _toolbox()

    result = json.loads(
        box.call("compare_spec", {"field": "정격 전류", "documents": ["G100", "M100"]})
    )

    assert retriever.calls == [
        ("정격 전류", ["ls-g100-v1"]),
        ("정격 전류", ["ls-m100-v1"]),
    ]
    assert set(result["문서별 근거"]) == {"G100", "M100"}


def test_비교_도구는_판정하지_않는다() -> None:
    """값이 다르다는 이유만으로 충돌로 단정하지 않는다는 규칙(DP-39)이
    도구 안에 숨으면 검증할 수 없다. 근거만 주고 판정은 모델이 한다."""
    box, _ = _toolbox()

    result = json.loads(
        box.call("compare_spec", {"field": "정격 전류", "documents": ["G100", "M100"]})
    )

    assert "판정" not in result
    assert "결론" not in result
    assert "조건과 단위를 먼저 확인" in result["안내"]


def test_비교_대상이_하나면_거부한다() -> None:
    box, retriever = _toolbox()

    result = json.loads(box.call("compare_spec", {"field": "정격 전류", "documents": ["G100"]}))

    assert "둘 이상" in result["오류"]
    assert retriever.calls == []


def test_일부_문서만_없으면_찾은_것과_못_찾은_것을_함께_준다() -> None:
    box, _ = _toolbox()

    result = json.loads(
        box.call("compare_spec", {"field": "정격 전류", "documents": ["G100", "없는제품"]})
    )

    assert "G100" in result["문서별 근거"]
    assert result["못 찾은 문서"] == ["없는제품"]


def test_모르는_도구는_예외_대신_알림이다() -> None:
    """모델이 없는 도구를 부르는 일이 있다. 예외로 끊으면 루프가 죽는다."""
    box, _ = _toolbox()

    result = json.loads(box.call("전혀_없는_도구", {}))

    assert "그런 도구가 없음" in result["오류"]


def test_근거_본문은_길이를_자른다() -> None:
    """도구 결과는 대화에 계속 쌓인다. 청크 전체를 넣으면 몇 번만 불러도 찬다."""
    from techdoc_rag.agent.tools import EVIDENCE_CHARS

    box, retriever = _toolbox()
    long_text = "가" * (EVIDENCE_CHARS * 3)

    def long_retrieve(question, document_ids=None):
        retriever.calls.append((question, document_ids))
        chunk = Chunk(
            chunk_id="x:0001",
            document_id="ls-g100-v1",
            document_version=1,
            page_start=1,
            page_end=1,
            text=long_text,
        )
        return RetrievalResult(
            chunks=[RetrievedChunk(chunk=chunk, score=0.9)], dropped_below_threshold=0
        )

    retriever.retrieve = long_retrieve
    result = json.loads(box.call("search_manual", {"question": "질문"}))

    assert len(result["근거"][0]["본문"]) == EVIDENCE_CHARS
