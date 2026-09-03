"""화면 표시 판단 테스트 (#29).

핵심은 하나다: answered=False인 응답의 text가 '답변' 자리로 나가지 않는 것.
HTTP 호출부는 실물 FastAPI(TestClient가 아니라 to_display 입력이 되는
JSON 계약)로 확인하므로 여기서는 변환만 본다.
"""

from __future__ import annotations

from techdoc_rag.ui.chat_view import DisplayCitation, to_display


def _citation(used: bool) -> dict:
    return {
        "document_id": "ls-m100-v1",
        "document_version": 1,
        "display_name": "M100 사용설명서.pdf",
        "page_start": 12,
        "page_end": 18,
        "chunk_id": "ls-m100-v1:0028",
        "is_used_in_answer": used,
    }


def test_답변된_응답은_텍스트와_근거_라벨이_나온다() -> None:
    display = to_display(
        {
            "text": "-10~50℃입니다 [1].",
            "citations": [_citation(True), _citation(False)],
            "no_answer_reason": None,
            "answered": True,
        }
    )

    assert display.answered
    assert display.text == "-10~50℃입니다 [1]."
    assert display.notice is None
    assert display.citations[0] == DisplayCitation(
        label="M100 사용설명서.pdf p.12~18", is_used_in_answer=True
    )
    assert display.citations[1].is_used_in_answer is False


def test_NOT_GROUNDED의_원문은_답변_자리로_나가지_않는다() -> None:
    """Answer 계약: 근거 사용이 확인되지 않은 원문을 답변처럼 노출하면
    근거 없는 내용이 출처 있는 답으로 오인된다. 원문은 접힌 상자용 필드로만."""
    display = to_display(
        {
            "text": "아마 10A일 것입니다.",
            "citations": [_citation(False)],
            "no_answer_reason": "NOT_GROUNDED",
            "answered": False,
        }
    )

    assert display.answered is False
    assert display.text is None  # 답변 자리는 비어야 한다
    assert display.notice is not None and "확인되지 않아" in display.notice
    assert display.ungrounded_text == "아마 10A일 것입니다."


def test_검색_실패_이유들은_원문_없이_안내문만_나온다() -> None:
    for reason in ("NO_RELEVANT_CHUNK", "LOW_RELEVANCE"):
        display = to_display(
            {
                "text": "등록된 문서에서 관련 근거를 찾지 못했습니다.",
                "citations": [],
                "no_answer_reason": reason,
                "answered": False,
            }
        )
        assert display.text is None
        assert display.ungrounded_text is None  # 원문 상자도 없다 — LLM을 안 불렀다
        assert display.notice


def test_모르는_이유값도_화면이_깨지지_않는다() -> None:
    """서버가 새 이유를 추가해도 구버전 화면이 스택트레이스를 내면 안 된다."""
    display = to_display(
        {"text": "원문", "citations": [], "no_answer_reason": "NEW_REASON", "answered": False}
    )

    assert display.text is None
    assert display.notice is not None and "NEW_REASON" in display.notice


def test_같은_페이지는_물결_없이_표시된다() -> None:
    citation = _citation(True)
    citation["page_end"] = citation["page_start"]

    display = to_display(
        {"text": "답 [1]", "citations": [citation], "no_answer_reason": None, "answered": True}
    )

    assert display.citations[0].label == "M100 사용설명서.pdf p.12"
