"""화면이 쓸 API 호출과 표시 판단 (#29).

Streamlit 렌더링과 분리한 이유: 여기 있는 판단(answered=False면 text를
답변으로 내보내지 않는다, 이유별 안내문, 오류를 스택트레이스가 아니라
문구로)은 단위 테스트가 필요하고, 렌더링 자체는 실물 확인으로 충분하다.

파이프라인 코드를 import하지 않는다 — 화면은 FastAPI의 HTTP 계약만 안다.
경계를 우회하면 #27의 오류 매핑·길이 상한이 전부 무의미해진다.

**여기서 나가는 예외는 ApiError뿐이어야 한다.** 사이드바의 상태 조회는
모듈 최상위에서 실행되므로, 다른 예외가 새면 사이드바가 아니라 페이지
전체가 죽는다(리뷰 #30 M3).

이 보증은 "이 모듈의 공개 함수에 어떤 응답이 와도"까지다 — 주소 오설정,
서버 버전 불일치, JSON이 아닌 응답, 잘린 본문, 깊이 중첩·거대 정수 응답을
변형 테스트로 확인했다. to_display에 dict가 아닌 값을 직접 넣는 것처럼
이 모듈을 계약 밖으로 부르는 경우는 범위가 아니다.
"""

from __future__ import annotations

import http.client
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
    """API의 /chat 응답 JSON을 화면 표시용으로 바꾼다.

    응답 모양이 계약과 다르면 ApiError로 바꾼다. KeyError·TypeError를 그대로
    올리면 호출부가 ApiError만 잡고 있어 대화 영역에 스택트레이스가 뜬다
    (리뷰 #30 재검토). 서버 버전이 어긋났을 때 실제로 도달하는 경로다.
    """
    try:
        citations = [
            DisplayCitation(
                label=_page_label(citation), is_used_in_answer=citation["is_used_in_answer"]
            )
            for citation in response.get("citations") or []
        ]
        answered = response["answered"]
        text = response["text"]

        if answered:
            return DisplayAnswer(
                answered=True, text=text, notice=None, ungrounded_text=None, citations=citations
            )
        # no-answer 분기도 try 안에 둔다. reason이 문자열이 아니면 dict 조회에서
        # TypeError가 나는데, 서버 버전 불일치는 이 함수가 든 전제 그대로다.
        reason = response.get("no_answer_reason") or ""
        return DisplayAnswer(
            answered=False,
            text=None,
            notice=_NO_ANSWER_NOTICES.get(reason, f"답변을 확인할 수 없습니다 ({reason})."),
            ungrounded_text=text if reason == "NOT_GROUNDED" else None,
            citations=citations,
        )
    except (KeyError, TypeError) as error:
        raise ApiError(
            "API 응답 형식이 예상과 다릅니다. 서버와 화면의 버전이 맞는지 확인하세요."
        ) from error


@dataclass(frozen=True, slots=True)
class DisplayStep:
    """도구 한 번 호출. label은 화면에 한 줄로 접어 보일 제목이다."""

    label: str
    result: str


@dataclass(frozen=True, slots=True)
class DisplayAgentAnswer:
    text: str
    steps: list[DisplayStep]
    # 상한에 걸려 끊겼는지. 화면은 이때 "덜 찾은 답"이라고 알려야 한다.
    stopped_at_limit: bool
    evidence_pages: list[str] = field(default_factory=list)


def to_agent_display(response: dict) -> DisplayAgentAnswer:
    """API의 /agent 응답 JSON을 화면 표시용으로 바꾼다.

    to_display와 같은 계약이다 — 응답 모양이 다르면 ApiError로 바꾼다.
    여기는 answered 분기가 없다. 에이전트는 근거를 못 찾으면 그렇게 적은
    문장을 답으로 내고, 화면은 그것을 그대로 보인다.
    """
    try:
        steps = [
            DisplayStep(label=_step_label(step), result=str(step["result"]))
            for step in response.get("steps") or []
        ]
        return DisplayAgentAnswer(
            text=response["text"],
            steps=steps,
            stopped_at_limit=bool(response["stopped_at_limit"]),
            evidence_pages=[str(page) for page in response.get("evidence_pages") or []],
        )
    except (KeyError, TypeError) as error:
        raise ApiError(
            "API 응답 형식이 예상과 다릅니다. 서버와 화면의 버전이 맞는지 확인하세요."
        ) from error


def _step_label(step: dict) -> str:
    """`compare_spec(주위 온도, ['G100', 'M100'])` 처럼 인자까지 한 줄로 적는다.

    도구 이름만 적으면 같은 도구를 여러 번 부른 기록이 전부 같은 줄로 보여서
    무엇이 달랐는지 알 수 없다.
    """
    arguments = step["arguments"]
    if not isinstance(arguments, dict):
        raise TypeError(f"arguments가 dict가 아님: {type(arguments).__name__}")
    if not arguments:
        return f"{step['tool']}()"
    inside = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{step['tool']}({inside})"


def _page_label(citation: dict) -> str:
    pages = (
        f"p.{citation['page_start']}"
        if citation["page_start"] == citation["page_end"]
        else f"p.{citation['page_start']}~{citation['page_end']}"
    )
    return f"{citation['display_name']} {pages}"


def ask_api(base_url: str, path: str, question: str, timeout_seconds: float) -> dict:
    """질문을 POST한다. 실패는 화면에 보여줄 문구를 담은 ApiError로 바꾼다.

    path는 "chat" 또는 "agent"다. 감싸는 함수를 따로 두지 않고 인자로 받는다 —
    두 경로가 요청 본문도 오류 매핑도 같아서, 나누면 부르는 곳만 늘고
    어느 경로로 나가는지는 오히려 덜 보인다.
    """
    try:
        # 본문 조립과 Request 생성도 try 안에 둔다. 주소에 scheme이 없으면 여기서
        # ValueError가, 질문에 짝 없는 서로게이트가 있으면 인코딩에서
        # UnicodeEncodeError가 난다. 밖에 두면 그것만 예외 변환을 비껴간다.
        payload = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _endpoint(base_url, path),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return _decode(response.read())
    except urllib.error.HTTPError as error:
        detail = str(_read_json(error).get("detail") or "상세 없음")[:300]
        if error.code == 422:
            raise ApiError(f"질문이 서버에서 거부됐습니다: {detail}") from error
        if error.code == 503:
            raise ApiError(f"서버 구성요소 장애입니다: {detail}") from error
        raise ApiError(f"서버 오류 (HTTP {error.code}): {detail}") from error
    except (OSError, http.client.HTTPException, ValueError) as error:
        # URLError·TimeoutError는 OSError 하위라 따로 적지 않는다. HTTPException은
        # OSError가 아니어서 별도로 적는다 — Content-Length가 본문보다 큰 응답의
        # IncompleteRead가 여기로 온다. ValueError는 주소 오설정에서 온다
        # (scheme 없음, 빈 문자열, 개행 섞임 — UnicodeEncodeError도 하위다).
        raise ApiError(
            "API 서버에 연결할 수 없습니다. 주소와 서버 상태를 확인하세요 "
            f"({base_url!r})"
        ) from error


def fetch_health(base_url: str, timeout_seconds: float = 5.0) -> dict:
    """GET /health. 503(degraded)도 구성요소별 상태를 담아 돌려준다.

    components가 dict인 것까지 확인한다. 사이드바는 페이지 최상위에서
    이 결과를 순회하므로, 모양이 다르면 AttributeError가 나면서 사이드바가
    아니라 페이지 전체가 죽는다(리뷰 #30 재검토에서 실측으로 확인된 경로).
    """
    try:
        with urllib.request.urlopen(
            _endpoint(base_url, "health"), timeout=timeout_seconds
        ) as response:
            return _as_health(_decode(response.read()))
    except urllib.error.HTTPError as error:
        # 503의 detail 안에 200과 같은 모양(status/components)이 들어 있다.
        detail = _read_json(error).get("detail")
        if isinstance(detail, dict):
            return _as_health(detail)
        raise ApiError(f"상태 확인 실패 (HTTP {error.code})") from error
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise ApiError(f"API 서버에 연결할 수 없습니다 ({base_url!r}).") from error


def _as_health(body: dict) -> dict:
    """components가 항상 dict인 상태 응답으로 만든다.

    None을 그냥 통과시키면 안 된다. 사이드바의 `health.get("components", {})`는
    키가 있고 값이 None이면 기본값이 아니라 None을 돌려주고, 거기서
    AttributeError가 나면서 페이지 전체가 죽는다 — 앞서 고친 list 경우와
    실패 모드가 같다(리뷰 #30 3차). 검증기가 허용하는 값은 소비자가
    다룰 수 있는 값이어야 한다.
    """
    components = body.get("components")
    if components is None:
        components = {}
    if not isinstance(components, dict):
        raise ApiError(
            f"상태 응답 형식이 예상과 다릅니다: components가 {type(components).__name__}"
        )
    return {**body, "components": components}


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
    except (ValueError, RecursionError) as error:
        # JSONDecodeError·UnicodeDecodeError는 ValueError 하위이고, 4,300자리를
        # 넘는 정수 리터럴도 평범한 ValueError로 온다. RecursionError는
        # RuntimeError 하위라 따로 적어야 한다 — 깊게 중첩된 응답이면 여기서 나고,
        # 그대로 새면 사이드바에서 페이지 전체가 죽는다(리뷰 #30 4차).
        # 몇 겹부터 나는지는 플랫폼이 정한다. 같은 깊이 3,000이 Windows/3.13에서는
        # RecursionError를 내고 Linux/3.12에서는 그대로 파싱됐다(CI 34194719381).
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
    except (ValueError, RecursionError, OSError, http.client.HTTPException):
        # 이 함수는 HTTPError 핸들러 '안에서' 불린다 — 여기서 예외가 나면
        # 바깥 except가 받지 못하고 그대로 샌다. 오류 본문은 부가 정보이므로
        # 읽기 실패는 빈 dict로 삼킨다(리뷰 #30 4차).
        return {}
    return body if isinstance(body, dict) else {}
