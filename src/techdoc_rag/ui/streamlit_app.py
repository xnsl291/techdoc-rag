"""질의 화면 (#29, #42). UC-1(Manual QA) 데모.

질의 방식이 둘이다. **일반 질의**는 검색 한 번에 답 하나이고, 인용 번호로
답변이 근거를 실제로 썼는지 확인한다(DP-56). **에이전트**는 모델이 도구를
골라 여러 번 찾고, 그 호출 기록을 화면에 펼쳐 보인다 — 제품 간 비교처럼
몇 번을 어떤 순서로 찾을지가 질문마다 다른 경우에 쓴다(#42).

실행 (FastAPI가 먼저 떠 있어야 함):
    uvicorn --factory techdoc_rag.api.app:create_default_app \
        --host 127.0.0.1 --port 8000 --app-dir src
    python scripts/run_ui.py

런처를 쓰는 이유: 주소를 코드에 박아 실행 위치·명령 실수와 무관하게
127.0.0.1에만 열기 위함이다. .streamlit/config.toml에도 같은 값이 있지만
Streamlit은 그 파일을 실행 디렉터리 기준으로 찾으므로, 저장소 루트가 아닌
곳에서 띄우면 기본값 0.0.0.0으로 조용히 되돌아간다(리뷰 #30 M1).
직접 실행할 때는 --server.address=127.0.0.1을 반드시 붙일 것.

판단 로직은 chat_view.py에 있고 여기는 렌더링만 한다. 지난 문답은
session_state에만 쌓인다(DP-44 — 서버는 stateless, 새로고침하면 사라짐).
공개 배포하지 않는다(07 §9.5 저작권 검토) — 화면 녹화·스크린샷 용도다.
"""

from __future__ import annotations

import os

import streamlit as st

from techdoc_rag.ui.chat_view import (
    ApiError,
    DisplayAgentAnswer,
    DisplayAnswer,
    ask_api,
    fetch_health,
    to_agent_display,
    to_display,
)

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
        components = health.get("components", {})
        # 전체 판정을 먼저 보인다. 구성요소만 나열하면 목록이 비었을 때
        # 정상인지 조회를 못 한 것인지 화면에서 구분되지 않는다.
        if health.get("status") == "ok":
            st.success("정상")
        else:
            st.warning(f"이상 있음 ({health.get('status', '알 수 없음')})")
        for name, state in components.items():
            icon = "🟢" if state == "ok" else "🔴"
            st.write(f"{icon} {name}: {state}")
        if not components:
            st.caption("보고된 구성요소가 없습니다.")
    except ApiError as error:
        st.error(str(error))

    st.divider()
    st.subheader("질의 방식")
    mode = st.radio(
        "질의 방식",
        options=["일반 질의", "에이전트"],
        captions=[
            "검색 한 번, 답 하나. 인용 번호로 근거 사용을 확인합니다.",
            "모델이 도구를 골라 여러 번 찾습니다. 제품 간 비교에 씁니다.",
        ],
        label_visibility="collapsed",
    )

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
    if not display.citations:
        return
    if not display.answered:
        # 답변이 없으면 "답변에 사용된 근거"라는 말이 성립하지 않는다. 지금은
        # 서버가 인용 0건일 때만 NOT_GROUNDED를 내지만 그 규칙은 HTTP 계약에
        # 없으므로, 화면이 서버 내부 불변식에 기대지 않게 한다(리뷰 #30 L4).
        with st.expander(f"검색된 근거 {len(display.citations)}건"):
            for citation in display.citations:
                st.markdown(f"- {citation.label}")
        return
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


def _render_agent_answer(display: DisplayAgentAnswer) -> None:
    """답변보다 호출 기록을 먼저 보인다.

    무엇을 근거로 답했는지가 답 자체만큼 중요하다. 에이전트는 순서를 모델이
    정하므로, 기록이 없으면 왜 그런 답이 나왔는지 되짚을 수 없다.
    """
    if display.stopped_at_limit:
        # 덜 찾고 끊긴 답을 다 찾은 답처럼 보이면 안 된다.
        st.warning("도구 호출 상한에 걸려 중간에 끊었습니다. 근거가 부족할 수 있습니다.")
    if display.steps:
        with st.expander(f"🔧 도구 호출 {len(display.steps)}회", expanded=True):
            for index, step in enumerate(display.steps, start=1):
                st.markdown(f"**{index}. `{step.label}`**")
                st.code(step.result, language="json")
    else:
        st.caption("도구를 부르지 않고 답했습니다.")
    st.markdown(display.text)
    if display.evidence_pages:
        # "답변에 사용된 근거"라고 쓰지 않는다. 도구가 가져온 것일 뿐,
        # 답변에 실제로 쓰였는지는 이 경로에서 확인하지 않는다(DP-56 미적용).
        st.markdown("**도구가 가져온 근거**")
        for page in display.evidence_pages:
            st.markdown(f"- {page}")


def _render(result: object) -> None:
    if isinstance(result, DisplayAnswer):
        _render_answer(result)
    elif isinstance(result, DisplayAgentAnswer):
        _render_agent_answer(result)
    else:
        st.error(result)


for past_question, past_result in st.session_state.history:
    with st.chat_message("user"):
        st.write(past_question)
    with st.chat_message("assistant"):
        _render(past_result)

question = st.chat_input("매뉴얼에 대해 질문하세요")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        agent_mode = mode == "에이전트"
        spinner = "도구를 골라 찾는 중..." if agent_mode else "근거를 찾고 답을 만드는 중..."
        with st.spinner(spinner):
            try:
                path = "agent" if agent_mode else "chat"
                body = ask_api(API_BASE_URL, path, question, REQUEST_TIMEOUT_SECONDS)
                display: object = to_agent_display(body) if agent_mode else to_display(body)
            except ApiError as error:
                display = str(error)
        _render(display)
    st.session_state.history.append((question, display))
