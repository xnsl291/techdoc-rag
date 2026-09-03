"""화면이 쓸 API 호출과 표시 판단 (#29).

Streamlit 렌더링과 분리한 이유: 여기 있는 판단(answered=False면 text를
답변으로 내보내지 않는다, 이유별 안내문, 오류를 스택트레이스가 아니라
문구로)은 단위 테스트가 필요하고, 렌더링 자체는 실물 확인으로 충분하다.

파이프라인 코드를 import하지 않는다 — 화면은 FastAPI의 HTTP 계약만 안다.
경계를 우회하면 #27의 오류 매핑·길이 상한이 전부 무의미해진다.

**여기서 나가는 예외는 ApiError뿐이어야 한다.** 사이드바의 상태 조회는
모듈 최상위에서 실행되므로, 다른 예외가 새면 사이드바가 아니라 페이지
전체가 죽는다(리뷰 #30 M3).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_NO_ANSWER_NOTICES = {
    "NO_RELEVANT_CHUNK": "등록된 문서에서 관련 근거를 찾지 못했습니다.",
    "LOW_RELEVANCE": "근거 후보는 있었지만 관련도가 기준에 미치지 못했습니다.",
    "NOT_GROUNDED": "답변이 생성됐지만 근거 사용이 확인되지 않아 표시하지 않습니다.",
}


class ApiError(Exception):
    """화면에 그대로 보여줄 수 있는 문구를 담는다."""


@dataclass(frozen=True, slots=True)
class DisplayCitation:
    label: str  # "문서명 p.10~11"
    is_used_in_answer: bool


@dataclass(frozen=True, slots=True)
class DisplayAnswer:
    answered: bool
    text: str | None  # 답변으로 보여도 되는 텍스트. answered=False면 None
    notice: str | None  # No-answer 안내문
    # NOT_GROUNDED의 LLM 원문. 경고 라벨을 단 접힌 상자로만 보여준다(승인 2026-09-03)
    # — 답변처럼 노출하면 근거 없는 내용이 출처 있는 답으로 오인된다(Answer 계약).
    ungrounded_text: str | None
    citations: list[DisplayCitation] = field(default_factory=list)


def to_display(response: dict) -> DisplayAnswer:
    """API의 /chat 응답 JSON을 화면 표시용으로 바꾼다."""
    citations = [
        DisplayCitation(
            label=_page_label(citation), is_used_in_answer=citation["is_used_in_answer"]
        )
        for citation in response.get("citations", [])
    ]
    if response["answered"]:
        return DisplayAnswer(
            answered=True,
            text=response["text"],
            notice=None,
            ungrounded_text=None,
            citations=citations,
        )
    reason = response.get("no_answer_reason") or ""
    return DisplayAnswer(
        answered=False,
        text=None,
        notice=_NO_ANSWER_NOTICES.get(reason, f"답변을 확인할 수 없습니다 ({reason})."),
        ungrounded_text=response["text"] if reason == "NOT_GROUNDED" else None,
        citations=citations,
    )


def _page_label(citation: dict) -> str:
    pages = (
        f"p.{citation['page_start']}"
        if citation["page_start"] == citation["page_end"]
        else f"p.{citation['page_start']}~{citation['page_end']}"
    )
    return f"{citation['display_name']} {pages}"


def ask_api(base_url: str, question: str, timeout_seconds: float) -> dict:
    """POST /chat. 실패는 화면에 보여줄 문구를 담은 ApiError로 바꾼다."""
    payload = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(base_url, "chat"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return _decode(response.read())
    except urllib.error.HTTPError as error:
        detail = str(_read_json(error).get("detail") or "상세 없음")[:300]
        if error.code == 422:
            raise ApiError(f"질문이 서버에서 거부됐습니다: {detail}") from error
        if error.code == 503:
            raise ApiError(f"서버 구성요소 장애입니다: {detail}") from error
        raise ApiError(f"서버 오류 (HTTP {error.code}): {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ApiError(
            "API 서버에 연결할 수 없습니다. FastAPI가 떠 있는지 확인하세요 "
            f"({base_url})"
        ) from error


def fetch_health(base_url: str, timeout_seconds: float = 5.0) -> dict:
    """GET /health. 503(degraded)도 구성요소별 상태를 담아 돌려준다."""
    try:
        with urllib.request.urlopen(
            _endpoint(base_url, "health"), timeout=timeout_seconds
        ) as response:
            return _decode(response.read())
    except urllib.error.HTTPError as error:
        # 503의 detail 안에 200과 같은 모양(status/components)이 들어 있다.
        detail = _read_json(error).get("detail")
        if isinstance(detail, dict):
            return detail
        raise ApiError(f"상태 확인 실패 (HTTP {error.code})") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ApiError("API 서버에 연결할 수 없습니다.") from error


def _endpoint(base_url: str, path: str) -> str:
    # 끝 슬래시가 붙은 주소(환경변수로 흔히 들어온다)를 그대로 이으면 //chat이 되어
    # 404가 나고, 화면에는 URL 조합 실수가 아니라 서버 오류처럼 보인다.
    return f"{base_url.rstrip('/')}/{path}"


def _decode(raw: bytes) -> dict:
    """응답 본문을 dict로. JSON이 아니면 ApiError로 바꾼다.

    API가 아닌 다른 서버(HTML을 주는)를 가리켰을 때 JSONDecodeError가
    화면까지 올라가 스택트레이스가 뜨던 것을 막는다(리뷰 #30 M3).
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ApiError(
            "API 응답을 이해할 수 없습니다. 주소가 이 서비스의 것이 맞는지 확인하세요."
        ) from error
    if not isinstance(body, dict):
        raise ApiError(f"API 응답 형식이 예상과 다릅니다: {type(body).__name__}")
    return body


def _read_json(error: urllib.error.HTTPError) -> dict:
    """오류 본문을 dict로. 읽을 수 없으면 빈 dict — 여기서 또 실패하면 안 된다."""
    try:
        body = json.loads(error.read())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return body if isinstance(body, dict) else {}
