"""Ollama 임베딩 어댑터 (bge-m3, DP-49).

ports.EmbeddingModel 계약의 요구를 그대로 구현한다. 특히 두 가지가 실측에서 나왔다.

- 실효 입력 상한은 컨텍스트 길이가 아니라 배치 크기(num_batch)다. /api/show는
  8192, /api/ps는 4096을 주는데 실제로 잘리는 지점은 num_batch(기본 2048)였다.
  그래서 이 어댑터는 options.num_batch를 직접 지정하고 그 값을 max_input_tokens로
  돌려준다.
- localhost는 IPv6로 먼저 해석되는데 Ollama가 IPv4에만 바인딩되어 있어 요청당
  2초가 붙는다(2,162ms 대 86ms). 조용히 느려지는 대신 생성 시점에 거부한다.
"""

from __future__ import annotations

import http.client
import json
from collections.abc import Sequence
from urllib.parse import urlsplit

from techdoc_rag.domain.errors import IndexingError

_EMBED_PATH = "/api/embed"


class OllamaEmbeddingModel:
    """Ollama /api/embed를 호출한다. 커넥션은 인스턴스가 재사용한다."""

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        batch_size: int,
        num_batch: int,
        embedding_version: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        parts = urlsplit(endpoint)
        if parts.hostname == "localhost":
            raise IndexingError(
                "endpoint에 localhost를 쓰지 말 것. IPv6 폴백으로 요청당 2초가 붙는다. "
                "127.0.0.1을 쓸 것 (2026-08-30 실측: 2,162ms 대 86ms)"
            )
        if parts.hostname is None or parts.port is None:
            raise IndexingError(f"endpoint 형식이 잘못됨: {endpoint} (예: http://127.0.0.1:11434)")
        if batch_size <= 0 or num_batch <= 0:
            raise IndexingError("batch_size와 num_batch는 양수여야 함")

        self._model_name = model_name
        self._host = parts.hostname
        self._port = parts.port
        self._batch_size = batch_size
        self._num_batch = num_batch
        self._embedding_version = embedding_version
        self._timeout_seconds = timeout_seconds
        self._connection: http.client.HTTPConnection | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """실제 로드된 모델이 내는 벡터 길이.

        하드코딩하면 모델이 바뀌었을 때 오류 없이 값만 어긋난다(05 CR-06).
        아직 모르면 짧은 텍스트 하나를 실제로 임베딩해 잰다. I/O가 일어난다.
        """
        if self._dimension is None:
            self.embed_query("dimension probe")
        assert self._dimension is not None
        return self._dimension

    @property
    def embedding_version(self) -> str:
        return self._embedding_version

    @property
    def max_input_tokens(self) -> int:
        # 실효 상한은 컨텍스트가 아니라 num_batch다. 모듈 docstring 참조.
        return self._num_batch

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self._batch_size:
            # 여기서 몰래 나누면 호출자가 색인 리스를 갱신할 자리가 없다.
            # 나누는 것은 호출자의 일이다 (EmbeddingModel 계약).
            raise IndexingError(
                f"한 번의 호출은 한 배치다: {len(texts)}개 > batch_size {self._batch_size}. "
                f"호출자가 나눠서 보낼 것"
            )
        return self._request(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {
                "model": self._model_name,
                "input": texts,
                "options": {"num_batch": self._num_batch},
            }
        ).encode("utf-8")

        body = self._send_with_retry_once(payload)
        try:
            embeddings = body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise IndexingError(f"임베딩 응답에 embeddings가 없음: {str(body)[:200]}") from exc

        if len(embeddings) != len(texts):
            raise IndexingError(
                f"입력 {len(texts)}개에 벡터 {len(embeddings)}개가 옴. "
                f"순서 짝짓기가 불가능하므로 결과를 쓰지 않음"
            )
        # 검증을 전부 통과한 뒤에만 차원을 캐시에 커밋한다. 도중에 커밋하면
        # 혼입 배치를 거부하고도 오염된 차원이 남아 이후 정상 응답까지 거부한다.
        observed_dimension = self._dimension
        for vector in embeddings:
            if observed_dimension is None:
                observed_dimension = len(vector)
            elif len(vector) != observed_dimension:
                raise IndexingError(
                    f"벡터 차원 불일치: {len(vector)} != {observed_dimension}. "
                    f"모델이 바뀌었는지 확인할 것"
                )
        self._dimension = observed_dimension
        return embeddings

    def _send_with_retry_once(self, payload: bytes) -> dict:
        """보내고, 죽은 커넥션이면 한 번만 새로 열어 다시 보낸다.

        재사용 커넥션은 서버 재시작이나 keep-alive 만료로 조용히 죽는다.
        그 경우 한 번의 재연결은 오류가 아니다. 두 번째 실패는 올린다.

        전제: /api/embed는 서버 상태를 바꾸지 않는 멱등 연산이라 이중 전송이
        안전하다. 이 재시도 패턴을 비멱등 API에 복사하면 사고가 된다.
        """
        for attempt in (1, 2):
            try:
                connection = self._ensure_connection()
                connection.request(
                    "POST",
                    _EMBED_PATH,
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                raw = response.read()
                if response.status != 200:
                    raise IndexingError(
                        f"임베딩 요청 실패: HTTP {response.status} {raw[:200]!r}"
                    )
                return json.loads(raw)
            except IndexingError:
                raise
            except (http.client.HTTPException, ConnectionError, TimeoutError, OSError) as exc:
                self.close()
                if attempt == 2:
                    raise IndexingError(
                        f"임베딩 서버 접속 실패: {self._host}:{self._port} ({exc})"
                    ) from exc
            except json.JSONDecodeError as exc:
                raise IndexingError(f"임베딩 응답이 JSON이 아님: {exc}") from exc
        raise AssertionError("unreachable")

    def _ensure_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout_seconds
            )
        return self._connection
