"""화면 표시 판단·API 호출 테스트 (#29).

핵심은 둘이다. answered=False인 응답의 text가 '답변' 자리로 나가지 않는 것,
그리고 **여기서 나가는 예외가 ApiError뿐인 것** — 다른 예외가 새면 사이드바가
아니라 페이지 전체가 죽는다(리뷰 #30 M3).

호출부는 가짜 HTTP 서버로 본다. 앞선 커밋은 "실물 FastAPI로 확인한다"고
적어 뒀지만 그런 테스트가 없었고, 리뷰의 mutation(422 분기 무력화)이
그대로 통과해 공백이 드러났다(#30 M2).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from techdoc_rag.ui.chat_view import (
    ApiError,
    DisplayCitation,
    _endpoint,
    ask_api,
    fetch_health,
    to_display,
)


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


class _FakeApiHandler(BaseHTTPRequestHandler):
    """behavior에 따라 /chat·/health 응답을 흉내 내는 서버."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (표준 라이브러리 시그니처)
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(json.loads(self.rfile.read(length)))  # type: ignore[attr-defined]
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        behavior = self.server.behavior  # type: ignore[attr-defined]
        if behavior == "too_long":
            self._reply(422, json.dumps({"detail": "질문이 너무 김: 1001자"}))
        elif behavior == "down":
            self._reply(503, json.dumps({"detail": "LLM 서버 접속 실패"}))
        elif behavior == "teapot":
            self._reply(418, json.dumps({"detail": "알 수 없음"}))
        elif behavior == "html":
            self._reply(200, "<html>다른 서버입니다</html>")
        elif behavior == "short_body":
            # Content-Length가 실제 본문보다 큼 → 클라이언트가 IncompleteRead를 만난다.
            self.send_response(200)
            self.send_header("Content-Length", "9999")
            self.end_headers()
            self.wfile.write(b'{"partial')
            self.close_connection = True
        else:
            self._reply(
                200,
                json.dumps(
                    {
                        "text": "답 [1].",
                        "citations": [],
                        "no_answer_reason": None,
                        "answered": True,
                    },
                    ensure_ascii=False,
                ),
            )

    def do_GET(self) -> None:  # noqa: N802
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        behavior = self.server.behavior  # type: ignore[attr-defined]
        payload = {"status": "ok", "components": {"sqlite": "ok"}}
        if behavior == "degraded":
            # FastAPI가 503으로 낼 때의 모양 — detail 안에 같은 구조가 들어 있다.
            degraded = {"status": "degraded", "components": {"ollama": "실패: 연결 거부"}}
            self._reply(503, json.dumps({"detail": degraded}, ensure_ascii=False))
        elif behavior == "html":
            self._reply(200, "<html>다른 서버입니다</html>")
        elif behavior == "bad_components":
            self._reply(200, json.dumps({"status": "ok", "components": ["sqlite", "llm"]}))
        elif behavior == "null_components":
            self._reply(200, json.dumps({"status": "ok", "components": None}))
        elif behavior == "degraded_bad_components":
            broken = {"status": "degraded", "components": ["ollama"]}
            self._reply(503, json.dumps({"detail": broken}))
        elif behavior == "short_body":
            self.send_response(200)
            self.send_header("Content-Length", "9999")
            self.end_headers()
            self.wfile.write(b'{"partial')
            self.close_connection = True
        else:
            self._reply(200, json.dumps(payload, ensure_ascii=False))

    def _reply(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture()
def fake_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeApiHandler)
    server.behavior = "ok"  # type: ignore[attr-defined]
    server.bodies = []  # type: ignore[attr-defined]
    server.paths = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _base_url(server, trailing_slash: bool = False) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}" + ("/" if trailing_slash else "")


def test_질문이_JSON으로_전송되고_응답이_dict로_온다(fake_api) -> None:
    body = ask_api(_base_url(fake_api), "정격 전류는?", timeout_seconds=5)

    assert body["answered"] is True
    assert fake_api.bodies[-1] == {"question": "정격 전류는?"}
    assert fake_api.paths[-1] == "/chat"


def test_끝_슬래시가_있어도_경로가_겹치지_않는다() -> None:
    """//chat이 되면 404가 나고 화면에는 URL 조합 실수가 서버 오류로 보인다.

    가짜 서버의 self.path로는 검증할 수 없다 — http.server가 //chat을 /chat으로
    정규화해서, rstrip을 지워도 통과하는 공회전 테스트가 된다(리뷰 #30 재검토).
    조합 결과를 직접 본다.
    """
    assert _endpoint("http://127.0.0.1:8000/", "chat") == "http://127.0.0.1:8000/chat"
    assert _endpoint("http://127.0.0.1:8000", "chat") == "http://127.0.0.1:8000/chat"
    assert _endpoint("http://127.0.0.1:8000///", "health") == "http://127.0.0.1:8000/health"


def test_422는_질문_거부_문구가_된다(fake_api) -> None:
    fake_api.behavior = "too_long"

    with pytest.raises(ApiError, match="거부"):
        ask_api(_base_url(fake_api), "가" * 20, timeout_seconds=5)


def test_503은_장애_문구가_된다(fake_api) -> None:
    fake_api.behavior = "down"

    with pytest.raises(ApiError, match="장애"):
        ask_api(_base_url(fake_api), "질문", timeout_seconds=5)


def test_그밖의_상태코드도_ApiError다(fake_api) -> None:
    fake_api.behavior = "teapot"

    with pytest.raises(ApiError, match="418"):
        ask_api(_base_url(fake_api), "질문", timeout_seconds=5)


def test_JSON이_아닌_응답도_ApiError다(fake_api) -> None:
    """다른 서버를 가리켰을 때 JSONDecodeError가 화면까지 올라가면
    스택트레이스가 뜬다(리뷰 #30 M3)."""
    fake_api.behavior = "html"

    with pytest.raises(ApiError, match="이해할 수 없습니다"):
        ask_api(_base_url(fake_api), "질문", timeout_seconds=5)


def test_서버가_없으면_연결_안내_문구다() -> None:
    with pytest.raises(ApiError, match="연결할 수 없습니다"):
        ask_api("http://127.0.0.1:9", "질문", timeout_seconds=2)


def test_health는_정상과_degraded를_같은_모양으로_돌려준다(fake_api) -> None:
    assert fetch_health(_base_url(fake_api))["status"] == "ok"

    fake_api.behavior = "degraded"
    degraded = fetch_health(_base_url(fake_api))

    assert degraded["status"] == "degraded"
    assert degraded["components"]["ollama"].startswith("실패")


def test_health도_ApiError_외의_예외를_내보내지_않는다(fake_api) -> None:
    """사이드바 조회는 페이지 최상위에서 돈다 — 여기서 새면 페이지 전체가 죽는다."""
    fake_api.behavior = "html"
    with pytest.raises(ApiError):
        fetch_health(_base_url(fake_api))

    with pytest.raises(ApiError):
        fetch_health("http://127.0.0.1:9", timeout_seconds=2)


def test_스키마가_어긋난_응답은_ApiError다() -> None:
    """서버 버전이 어긋나면 KeyError·TypeError가 나는데, 호출부는 ApiError만
    잡으므로 대화 영역에 스택트레이스가 뜬다(리뷰 #30 재검토)."""
    with pytest.raises(ApiError, match="형식"):
        to_display({"message": "예상 못 한 모양"})  # answered·text 없음

    with pytest.raises(ApiError, match="형식"):
        to_display({"text": "답", "answered": True, "citations": ["문자열 근거"]})


def test_상태_응답의_components가_dict가_아니면_ApiError다(fake_api) -> None:
    """사이드바는 페이지 최상위에서 이 결과를 순회한다 — AttributeError가 나면
    사이드바가 아니라 페이지 전체가 죽는다(리뷰 #30 재검토에서 실측된 경로)."""
    fake_api.behavior = "bad_components"

    with pytest.raises(ApiError, match="components"):
        fetch_health(_base_url(fake_api))


def test_본문이_잘린_응답도_ApiError다(fake_api) -> None:
    """Content-Length가 본문보다 크면 IncompleteRead가 나는데, 이것은
    OSError 하위가 아니라 except 절을 그냥 빠져나갔다."""
    fake_api.behavior = "short_body"

    with pytest.raises(ApiError):
        ask_api(_base_url(fake_api), "질문", timeout_seconds=5)


def test_components가_null이면_빈_dict로_정규화된다(fake_api) -> None:
    """검증기가 허용하는 값은 소비자가 다룰 수 있어야 한다. 사이드바의
    health.get("components", {})는 값이 None이면 기본값이 아니라 None을 주고,
    거기서 AttributeError가 나며 페이지 전체가 죽는다(리뷰 #30 3차)."""
    fake_api.behavior = "null_components"

    health = fetch_health(_base_url(fake_api))

    assert health["components"] == {}
    health["components"].items()  # 소비자가 하는 그대로 — 여기서 터지면 안 된다


def test_degraded_응답의_components도_검사한다(fake_api) -> None:
    """503의 detail 안에도 같은 모양이 들어오므로 200 경로만 막으면 반쪽이다."""
    fake_api.behavior = "degraded_bad_components"

    with pytest.raises(ApiError, match="components"):
        fetch_health(_base_url(fake_api))


def test_health도_본문이_잘리면_ApiError다(fake_api) -> None:
    fake_api.behavior = "short_body"

    with pytest.raises(ApiError):
        fetch_health(_base_url(fake_api))


def test_no_answer_이유가_문자열이_아니어도_ApiError다() -> None:
    """서버 버전 불일치는 이 함수가 든 전제 그대로다. 하반부만 try 밖에 있으면
    dict 조회에서 TypeError가 그대로 샌다(리뷰 #30 3차)."""
    with pytest.raises(ApiError, match="형식"):
        to_display(
            {"text": "원문", "citations": [], "no_answer_reason": {"code": 1}, "answered": False}
        )


@pytest.mark.parametrize("base_url", ["", "myapi", "http://127.0.0.1:8000\n"])
def test_주소가_잘못돼도_ApiError다(base_url: str) -> None:
    """TECHDOC_API_URL 오설정에서 ValueError 계열이 그대로 새던 경로.
    빈 문자열은 현실적이다 — 환경변수를 빈 값으로 두면 기본값이 안 쓰인다."""
    with pytest.raises(ApiError):
        ask_api(base_url, "질문", timeout_seconds=2)

    with pytest.raises(ApiError):
        fetch_health(base_url, timeout_seconds=2)
