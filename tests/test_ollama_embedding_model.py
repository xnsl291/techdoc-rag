"""OllamaEmbeddingModel 테스트.

실제 Ollama 없이 가짜 HTTP 서버로 계약을 검증한다. 실물 검증(차원 1024,
num_batch 잘림 지점)은 scripts/verify_embedding_real.py가 담당한다.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel
from techdoc_rag.domain.errors import IndexingError


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """입력 텍스트마다 그 텍스트에서 유도한 벡터를 돌려준다.

    벡터의 첫 성분이 입력 순번이 아니라 입력 '내용'에서 나오므로,
    순서가 뒤바뀌면 테스트가 그것을 잡는다.
    """

    server_version = "FakeOllama/1"
    # HTTP/1.0이면 응답마다 연결이 닫혀 커넥션 재사용 테스트가 성립하지 않는다.
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (표준 라이브러리 시그니처)
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)  # type: ignore[attr-defined]
        self.server.connections.add(self.client_address[1])  # type: ignore[attr-defined]

        behavior = self.server.behavior  # type: ignore[attr-defined]
        if behavior == "http_500":
            self._reply(500, b"boom")
            return
        if behavior == "not_json":
            self._reply(200, b"this is not json")
            return
        if behavior == "wrong_count":
            embeddings = [[1.0, 0.0]]
        elif behavior == "wrong_dimension":
            embeddings = [[float(sum(text.encode()) % 97), 0.0, 0.0][: 3 if i == 0 else 2]
                          for i, text in enumerate(request["input"])]
        else:
            embeddings = [
                [float(sum(text.encode()) % 97), 1.0] for text in request["input"]
            ]
        self._reply(200, json.dumps({"embeddings": embeddings}).encode())

    def _reply(self, status: int, body: bytes) -> None:
        # HTTP/1.1 keep-alive에는 Content-Length가 필수다. 없으면 연결이 끊긴다.
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.connections = set()  # type: ignore[attr-defined]
    server.behavior = "ok"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _model(server, batch_size: int = 8, num_batch: int = 2048) -> OllamaEmbeddingModel:
    return OllamaEmbeddingModel(
        model_name="bge-m3",
        endpoint=f"http://127.0.0.1:{server.server_address[1]}",
        batch_size=batch_size,
        num_batch=num_batch,
        embedding_version="v1-test",
    )


def test_입력_순서와_출력_순서가_같다(fake_server) -> None:
    texts = ["가나다", "라마바", "사아자", "차카타"]
    model = _model(fake_server)

    vectors = model.embed_documents(texts)

    expected_heads = [float(sum(text.encode()) % 97) for text in texts]
    assert [vector[0] for vector in vectors] == expected_heads


def test_같은_텍스트는_같은_벡터가_나온다(fake_server) -> None:
    model = _model(fake_server)

    first = model.embed_query("정격출력")
    second = model.embed_query("정격출력")

    assert first == second


def test_여러_번_호출해도_커넥션이_새로_열리지_않는다(fake_server) -> None:
    model = _model(fake_server)

    for _ in range(5):
        model.embed_query("반복 호출")

    # 서버가 관측한 클라이언트 출발지 포트가 하나면 커넥션이 재사용된 것이다.
    assert len(fake_server.connections) == 1


def test_배치_크기를_넘는_호출은_거부한다(fake_server) -> None:
    model = _model(fake_server, batch_size=2)

    with pytest.raises(IndexingError, match="한 번의 호출은 한 배치"):
        model.embed_documents(["a", "b", "c"])


def test_num_batch를_요청에_명시한다(fake_server) -> None:
    model = _model(fake_server, num_batch=4096)

    model.embed_query("옵션 확인")

    request = fake_server.requests[-1]
    assert request["options"]["num_batch"] == 4096
    assert model.max_input_tokens == 4096


def test_dimension은_실제_응답에서_재고_캐시한다(fake_server) -> None:
    model = _model(fake_server)

    assert model.dimension == 2  # 가짜 서버는 2차원을 돌려준다
    request_count = len(fake_server.requests)
    assert model.dimension == 2  # 두 번째 접근은 프로브를 다시 하지 않는다
    assert len(fake_server.requests) == request_count


def test_응답_개수가_입력과_다르면_IndexingError(fake_server) -> None:
    fake_server.behavior = "wrong_count"
    model = _model(fake_server)

    with pytest.raises(IndexingError, match="순서 짝짓기"):
        model.embed_documents(["a", "b", "c"])


def test_차원이_섞이면_IndexingError(fake_server) -> None:
    fake_server.behavior = "wrong_dimension"
    model = _model(fake_server)

    with pytest.raises(IndexingError, match="차원 불일치"):
        model.embed_documents(["a", "b"])


def test_HTTP_오류와_비JSON_응답은_IndexingError(fake_server) -> None:
    model = _model(fake_server)

    fake_server.behavior = "http_500"
    with pytest.raises(IndexingError, match="HTTP 500"):
        model.embed_query("오류")

    fake_server.behavior = "not_json"
    with pytest.raises(IndexingError, match="JSON이 아님"):
        model.embed_query("비정상")


def test_죽은_커넥션은_한_번만_재연결한다(fake_server) -> None:
    model = _model(fake_server)
    model.embed_query("워밍업")

    # 서버 쪽에서 커넥션이 끊긴 상황을 흉내 낸다.
    model._connection.close()  # noqa: SLF001 (커넥션 사망 재현에 내부 접근이 필요함)

    assert model.embed_query("재연결") is not None
    assert len(fake_server.connections) == 2  # 원래 1 + 재연결 1


def test_localhost는_생성_시점에_거부한다() -> None:
    with pytest.raises(IndexingError, match="localhost"):
        OllamaEmbeddingModel(
            model_name="bge-m3",
            endpoint="http://localhost:11434",
            batch_size=8,
            num_batch=2048,
            embedding_version="v1",
        )


def test_빈_입력은_빈_결과다(fake_server) -> None:
    assert _model(fake_server).embed_documents([]) == []
