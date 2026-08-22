"""교체 가능한 외부 구성요소의 계약.

NFR-003이 LLM, Embedding, Vector Store를 상위 기능 수정 없이 교체할 수 있어야 한다고 요구한다.
서비스 계층은 구현체가 아니라 이 Protocol에만 의존한다.

adapters 패키지의 구현체는 여기를 import하지 않는다. 구조적 타이핑이므로
시그니처만 맞으면 되고, 그 덕에 의존성이 한 방향으로만 흐른다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk


class EmbeddingModel(Protocol):
    """텍스트를 벡터로 변환한다.

    색인용과 질의용을 나눈 것은 비대칭 임베딩 모델 때문이다.
    일부 모델은 문서와 질의에 서로 다른 프리픽스를 요구한다.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """청크 벡터를 저장하고 검색한다.

    upsert는 document_id와 chunk_id로 결정론적 ID를 만들어 재실행해도
    중복 벡터가 생기지 않아야 한다.
    """

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        active_document_ids: Sequence[str],
    ) -> list[RetrievedChunk]: ...

    def delete_document(self, document_id: str) -> None: ...


class LlmClient(Protocol):
    """로컬 LLM 추론을 호출한다.

    생성 지연이 길어 스트리밍을 기본으로 둔다.
    구현체는 HTTP 커넥션을 재사용해야 한다. 요청마다 새로 열면 Windows에서
    TIME_WAIT가 쌓여 동적 포트가 고갈되고, 평가 스크립트가 문항을 연속 처리할 때 터진다.
    """

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]: ...
