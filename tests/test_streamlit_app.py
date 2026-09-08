"""화면 렌더링 테스트 (#42).

지금까지 streamlit_app.py는 "실물 확인으로 충분하다"고 두고 테스트가 없었다.
판단 로직을 chat_view로 뺐으니 남은 것은 렌더링뿐이라는 이유였는데, 렌더링에도
지켜야 할 것이 있다.

- 에이전트 모드가 /agent로 나가는가. 경로를 잘못 보내면 /chat이 답하고,
  화면에는 도구 기록 없는 답이 그냥 뜬다
- 도구 호출 기록이 실제로 화면에 나오는가. 이 기록을 보이는 것이 #42의 목적이다
- 근거 목록의 제목이 경로에 맞는가. 에이전트 경로에는 인용 번호 확인 장치가
  없으므로(DP-56 미적용) "답변에 사용된 근거"라고 쓰면 안 된다

AppTest는 화면 스크립트를 브라우저 없이 그대로 실행한다. API는 가짜로 바꾼다 —
여기서 보는 것은 화면이지 서버가 아니다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from techdoc_rag.ui import chat_view

APP = str(Path(__file__).resolve().parents[1] / "src" / "techdoc_rag" / "ui" / "streamlit_app.py")

AGENT_BODY = {
    "text": "G100은 중부하 -10~50℃, M100은 -10~50℃입니다.",
    "steps": [
        {"tool": "list_documents", "arguments": {}, "result": '{"문서": ["G100", "M100"]}'},
        {
            "tool": "compare_spec",
            "arguments": {"field": "주위 온도", "documents": ["G100", "M100"]},
            "result": '{"문서별 근거": {"G100": []}}',
        },
    ],
    "stopped_at_limit": False,
    "evidence_pages": ["ls-g100-v1 p.17~26", "ls-m100-v1 p.18~23"],
}

CHAT_BODY = {
    "text": "정격 전류는 5A입니다 [1].",
    "answered": True,
    "no_answer_reason": None,
    "citations": [
        {
            "document_id": "ls-g100-v1",
            "document_version": 1,
            "display_name": "G100 사용설명서.pdf",
            "page_start": 248,
            "page_end": 250,
            "chunk_id": "ls-g100-v1:0178",
            "is_used_in_answer": True,
        }
    ],
}


@pytest.fixture()
def app(monkeypatch):
    """API를 가짜로 바꾸고 호출된 경로를 기록한다."""
    calls: list[str] = []

    def fake_ask(base_url: str, path: str, question: str, timeout_seconds: float) -> dict:
        calls.append(path)
        return AGENT_BODY if path == "agent" else CHAT_BODY

    monkeypatch.setattr(chat_view, "ask_api", fake_ask)
    monkeypatch.setattr(
        chat_view, "fetch_health", lambda *_a, **_k: {"status": "ok", "components": {}}
    )
    test = AppTest.from_file(APP, default_timeout=30)
    test.run()
    assert not test.exception
    return test, calls


def _texts(test: AppTest) -> str:
    """화면에 나온 글을 한 덩어리로 모은다."""
    parts = [element.value for element in test.markdown]
    parts += [element.value for element in test.code]
    parts += [element.value for element in test.warning]
    parts += [element.value for element in test.caption]
    return "\n".join(str(part) for part in parts)


def test_기본은_일반_질의이고_chat으로_나간다(app) -> None:
    test, calls = app

    test.chat_input[0].set_value("정격 전류는?").run()

    assert calls == ["chat"]
    assert "답변에 사용된 근거" in _texts(test)


def test_에이전트를_고르면_agent로_나간다(app) -> None:
    """경로를 잘못 보내면 /chat이 답하고 화면에는 도구 기록 없는 답이 뜬다."""
    test, calls = app

    test.sidebar.radio[0].set_value("에이전트").run()
    test.chat_input[0].set_value("G100과 M100 비교해줘").run()

    assert calls == ["agent"]


def test_도구_호출_기록이_인자까지_화면에_나온다(app) -> None:
    """이 기록을 보이는 것이 #42의 목적이다."""
    test, _ = app

    test.sidebar.radio[0].set_value("에이전트").run()
    test.chat_input[0].set_value("G100과 M100 비교해줘").run()

    shown = _texts(test)
    assert [element.label for element in test.get("expander")] == ["🔧 도구 호출 2회"]
    assert "list_documents()" in shown
    assert "compare_spec(field='주위 온도', documents=['G100', 'M100'])" in shown
    assert '{"문서별 근거": {"G100": []}}' in shown  # 도구가 돌려준 결과 본문


def test_에이전트_근거는_사용됐다고_적지_않는다(app) -> None:
    """에이전트 경로에는 인용 번호로 사용 여부를 확인하는 장치가 없다(DP-56 미적용).
    일반 질의와 같은 문구를 쓰면 확인하지 않은 것을 확인한 것처럼 보인다."""
    test, _ = app

    test.sidebar.radio[0].set_value("에이전트").run()
    test.chat_input[0].set_value("비교해줘").run()

    shown = _texts(test)
    assert "도구가 가져온 근거" in shown
    assert "답변에 사용된 근거" not in shown
    assert "ls-g100-v1 p.17~26" in shown


def test_상한에_걸린_답은_경고를_함께_보인다(monkeypatch) -> None:
    """덜 찾고 끊긴 답이 다 찾은 답처럼 보이면 안 된다."""
    monkeypatch.setattr(
        chat_view,
        "ask_api",
        lambda *_a, **_k: {**AGENT_BODY, "stopped_at_limit": True},
    )
    monkeypatch.setattr(
        chat_view, "fetch_health", lambda *_a, **_k: {"status": "ok", "components": {}}
    )
    test = AppTest.from_file(APP, default_timeout=30)
    test.run()

    test.sidebar.radio[0].set_value("에이전트").run()
    test.chat_input[0].set_value("질문").run()

    assert any("상한" in str(element.value) for element in test.warning)


def test_API_오류는_스택트레이스가_아니라_문구로_나온다(monkeypatch) -> None:
    """여기서 나가는 예외가 ApiError뿐이라는 계약이 화면까지 이어지는지 본다."""

    def broken(*_a, **_k):
        raise chat_view.ApiError("API 서버에 연결할 수 없습니다.")

    monkeypatch.setattr(chat_view, "ask_api", broken)
    monkeypatch.setattr(
        chat_view, "fetch_health", lambda *_a, **_k: {"status": "ok", "components": {}}
    )
    test = AppTest.from_file(APP, default_timeout=30)
    test.run()

    test.sidebar.radio[0].set_value("에이전트").run()
    test.chat_input[0].set_value("질문").run()

    assert not test.exception
    assert any("연결할 수 없습니다" in str(element.value) for element in test.error)
