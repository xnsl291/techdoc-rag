"""OllamaLlmClient 테스트.

가짜 NDJSON 스트리밍 서버로 계약을 검증한다. 지난 리뷰(#23)의 교훈대로
취소·끊김은 서버 쪽에서 재현한다 — 클라이언트 쪽 close()는 http.client의
auto_open이 조용히 덮어 검증력이 없다.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from techdoc_rag.adapters.ollama_llm_client import OllamaLlmClient
from techdoc_rag.domain.errors import GenerationError


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    server_version = "FakeOllama/1"
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (표준 라이브러리 시그니처)
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)  # type: ignore[attr-defined]
        self.server.connections.add(self.client_address[1])  # type: ignore[attr-defined]

        behavior = self.server.behavior  # type: ignore[attr-defined]
        if behavior == "model_missing":
            body = json.dumps({"error": "model 'ghost' not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 스트리밍: chunked로 한 줄씩 내보낸다.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        if behavior == "empty":
            lines = [{"done": True}]
        elif behavior == "thinking":
            lines = [
                {"thinking": "생각 중...", "response": ""},
                {"response": "답"},
                {"response": "변"},
                {"done": True},
            ]
        elif behavior == "error_line":
            lines = [{"response": "일부"}, {"error": "runner crashed"}]
        else:
            lines = [
                {"response": "안"},
                {"response": "녕"},
                {"response": "하세요"},
                {"done": True},
            ]

        hold = self.server.hold_before_done  # type: ignore[attr-defined]
        for line in lines:
            # done 직전에 멈춰 "생성이 진행 중"인 상태를 만든다.
            if hold is not None and line.get("done") and not hold.wait(timeout=10):
                return
            self._write_chunk(json.dumps(line).encode() + b"\n")
        self._write_chunk(b"")  # chunked 종료

    def _write_chunk(self, data: bytes) -> None:
        try:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
        except (ConnectionError, OSError):
            # 클라이언트가 먼저 끊은 경우(취소). 서버가 죽을 일은 아니다.
            self.close_connection = True

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.connections = set()  # type: ignore[attr-defined]
    server.behavior = "ok"  # type: ignore[attr-defined]
    server.hold_before_done = None  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _client(
    server,
    max_concurrent: int = 1,
    queue_timeout: float = 5.0,
    generation_timeout: float = 10.0,
) -> OllamaLlmClient:
    return OllamaLlmClient(
        model_name="qwen3.5:4b-q4_K_M",
        endpoint=f"http://127.0.0.1:{server.server_address[1]}",
        temperature=0.0,
        runtime_context_tokens=8192,
        thinking_enabled=False,
        max_concurrent_generations=max_concurrent,
        queue_timeout_seconds=queue_timeout,
        generation_timeout_seconds=generation_timeout,
    )


def test_토큰이_조각_단위로_순서대로_나온다(fake_server) -> None:
    client = _client(fake_server)

    pieces = list(client.generate("질문", max_tokens=64))

    assert pieces == ["안", "녕", "하세요"]  # 한 덩어리가 아니라 세 조각


def test_요청에_옵션과_thinking_설정이_실린다(fake_server) -> None:
    client = _client(fake_server)

    list(client.generate("질문", max_tokens=64))

    request = fake_server.requests[-1]
    assert request["stream"] is True
    assert request["think"] is False
    assert request["options"]["num_predict"] == 64
    assert request["options"]["num_ctx"] == 8192


def test_thinking_조각은_출력에_섞이지_않는다(fake_server) -> None:
    fake_server.behavior = "thinking"
    client = _client(fake_server)

    pieces = list(client.generate("질문", max_tokens=64))

    assert pieces == ["답", "변"]


def test_완주하면_커넥션을_재사용한다(fake_server) -> None:
    client = _client(fake_server)

    for _ in range(3):
        list(client.generate("반복", max_tokens=64))

    assert len(fake_server.connections) == 1


def test_중간에_닫으면_커넥션을_버리고_다음_호출은_새로_연다(fake_server) -> None:
    client = _client(fake_server)

    stream = client.generate("취소될 질문", max_tokens=64)
    assert next(stream) == "안"
    stream.close()  # 사용자가 화면을 닫은 상황

    # 읽다 만 커넥션을 재사용했다면 다음 응답의 상태줄을 잘못 읽어 깨진다.
    pieces = list(client.generate("다음 질문", max_tokens=64))

    assert pieces == ["안", "녕", "하세요"]
    assert len(fake_server.connections) == 2  # 버리고 새로 열었음

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.completed is True  # 마지막 호출은 완주


def test_취소된_생성의_metrics는_미완주로_남는다(fake_server) -> None:
    client = _client(fake_server)

    stream = client.generate("취소", max_tokens=64)
    next(stream)
    stream.close()

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.completed is False
    assert metrics.first_token_seconds is not None


def test_동시_생성_한도를_넘는_요청은_대기하다_GenerationError(fake_server) -> None:
    hold = threading.Event()
    fake_server.hold_before_done = hold
    client = _client(fake_server, max_concurrent=1, queue_timeout=0.3)

    first = client.generate("느린 생성", max_tokens=64)
    next(first)  # 세마포어를 잡은 채 done 직전에서 멈춰 있음

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(client.generate("대기 초과", max_tokens=64)))
            with pytest.raises(GenerationError, match="대기 초과"):
                future.result(timeout=5)
    finally:
        hold.set()
        first.close()


def test_대기가_풀리면_다음_생성이_성공한다(fake_server) -> None:
    hold = threading.Event()
    fake_server.hold_before_done = hold
    client = _client(fake_server, max_concurrent=1, queue_timeout=5.0)

    first = client.generate("느린 생성", max_tokens=64)
    next(first)

    def finish_first_soon() -> None:
        hold.set()
        # 첫 생성을 소비 완료시켜 세마포어를 풀어준다.
        list(first)

    releaser = threading.Thread(target=finish_first_soon)
    releaser.start()
    fake_server.hold_before_done = None
    pieces = list(client.generate("두 번째", max_tokens=64))
    releaser.join(timeout=5)

    assert pieces == ["안", "녕", "하세요"]


def test_모델이_없으면_GenerationError(fake_server) -> None:
    fake_server.behavior = "model_missing"
    client = _client(fake_server)

    with pytest.raises(GenerationError, match="HTTP 404"):
        list(client.generate("질문", max_tokens=64))


def test_스트림_중간의_오류_줄은_GenerationError(fake_server) -> None:
    fake_server.behavior = "error_line"
    client = _client(fake_server)

    with pytest.raises(GenerationError, match="runner crashed"):
        list(client.generate("질문", max_tokens=64))


def test_빈_응답은_실패가_아니다(fake_server) -> None:
    fake_server.behavior = "empty"
    client = _client(fake_server)

    assert list(client.generate("질문", max_tokens=64)) == []

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.completed is True
    assert metrics.first_token_seconds is None  # 출력이 없었으니 첫 토큰도 없음


def test_지연이_대기와_생성으로_나뉘어_기록된다(fake_server) -> None:
    client = _client(fake_server)

    list(client.generate("질문", max_tokens=64))

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.wait_seconds >= 0
    assert metrics.first_token_seconds is not None
    assert metrics.total_seconds >= metrics.first_token_seconds


def test_localhost는_생성_시점에_거부한다() -> None:
    with pytest.raises(GenerationError, match="localhost"):
        OllamaLlmClient(
            model_name="m",
            endpoint="http://localhost:11434",
            temperature=0.0,
            runtime_context_tokens=8192,
            thinking_enabled=False,
            max_concurrent_generations=1,
            queue_timeout_seconds=1,
            generation_timeout_seconds=1,
        )
