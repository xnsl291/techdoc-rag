"""질의 화면 (#29). UC-1(Manual QA) 데모.

실행 (FastAPI가 먼저 떠 있어야 함):
    uvicorn --factory techdoc_rag.api.app:create_default_app \
        --host 127.0.0.1 --port 8000 --app-dir src
    streamlit run src/techdoc_rag/ui/streamlit_app.py

판단 로직은 chat_view.py에 있고 여기는 렌더링만 한다. 지난 문답은
session_state에만 쌓인다(DP-44 — 서버는 stateless, 새로고침하면 사라짐).
공개 배포하지 않는다(07 §9.5 저작권 검토) — 화면 녹화·스크린샷 용도다.
"""

from __future__ import annotations

import os

import streamlit as st

from techdoc_rag.ui.chat_view import ApiError, DisplayAnswer, ask_api, fetch_health, to_display

API_BASE_URL = os.getenv("TECHDOC_API_URL", "http://127.0.0.1:8000")
# 생성 상한(300초) + 대기 여유. API 쪽 타임아웃보다 짧으면 서버는 아직
# 생성 중인데 화면만 끊겨 "실패처럼 보이는 성공"이 된다.
REQUEST_TIMEOUT_SECONDS = 330

st.set_page_config(page_title="techdoc-rag", layout="wide")
st.title("기술문서 QA")
st.caption("등록된 매뉴얼에서 근거를 찾아 답합니다. 근거가 없으면 답하지 않습니다.")

with st.sidebar:
    st.subheader("서버 상태")
    try:
        health = fetch_health(API_BASE_URL)
        for name, state in health.get("components", {}).items():
            icon = "🟢" if state == "ok" else "🔴"
            st.write(f"{icon} {name}: {state}")
    except ApiError as error:
        st.error(str(error))

if "history" not in st.session_state:
    st.session_state.history = []  # (질문, DisplayAnswer | ApiError 문구)


def _render_answer(display: DisplayAnswer) -> None:
    if display.answered:
        st.markdown(display.text)
    else:
        st.warning(display.notice)
        if display.ungrounded_text:
            with st.expander("⚠️ 근거 미확인 원문 보기 (참고용 — 답변이 아님)"):
                st.text(display.ungrounded_text)
    if display.citations:
        used = [c for c in display.citations if c.is_used_in_answer]
        others = [c for c in display.citations if not c.is_used_in_answer]
        if used:
            st.markdown("**답변에 사용된 근거**")
            for citation in used:
                st.markdown(f"- 📌 {citation.label}")
        if others:
            with st.expander(f"검색됐지만 사용되지 않은 근거 {len(others)}건"):
                for citation in others:
                    st.markdown(f"- {citation.label}")


for past_question, past_result in st.session_state.history:
    with st.chat_message("user"):
        st.write(past_question)
    with st.chat_message("assistant"):
        if isinstance(past_result, DisplayAnswer):
            _render_answer(past_result)
        else:
            st.error(past_result)

question = st.chat_input("매뉴얼에 대해 질문하세요")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("근거를 찾고 답을 만드는 중..."):
            try:
                display: DisplayAnswer | str = to_display(
                    ask_api(API_BASE_URL, question, REQUEST_TIMEOUT_SECONDS)
                )
            except ApiError as error:
                display = str(error)
        if isinstance(display, DisplayAnswer):
            _render_answer(display)
        else:
            st.error(display)
    st.session_state.history.append((question, display))
