"""교체 가능한 외부 구성요소의 계약.

NFR-003이 LLM, Embedding, Vector Store를 상위 기능 수정 없이 교체할 수 있어야 한다고 요구한다.
서비스 계층은 구현체가 아니라 이 Protocol에만 의존한다.

adapters 패키지의 구현체는 여기를 import하지 않는다. 구조적 타이핑이므로
시그니처만 맞으면 되고, 그 덕에 의존성이 한 방향으로만 흐른다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

from techdoc_rag.domain.chunk import Chunk, RetrievedChunk
from techdoc_rag.domain.document import Document
from techdoc_rag.domain.indexing import IndexRun
from techdoc_rag.domain.parsing import ParsedDocument


class DocumentRepository(Protocol):
    """문서 생명주기의 정본을 관리한다(D-004, D-012).

    "logical_document_id당 active 버전은 1개"라는 불변식은 구현체가 코드가 아니라
    저장소 자체의 제약으로 강제해야 한다. 코드로만 지키면 버그 하나에 뚫리고,
    점검(03 §22.7)은 사후 발견일 뿐이다.

    활성 전환(activate)은 원자적이어야 한다. 신규 활성화와 구버전 비활성화
    사이에 중단돼도 active가 2개이거나 0개인 상태가 관측되면 안 된다(01 §17.6).

    재색인은 항상 새 버전을 등록해서 한다(DP-50). 같은 document_id를 제자리에서
    다시 색인하면 is_active가 1인 채로 검색에서 빠진다. mark_indexing이 이를 막는다.
    """

    def initialize(self) -> None: ...

    def register(self, document: Document) -> None: ...

    def get(self, document_id: str) -> Document | None: ...

    def find_by_sha256(self, logical_document_id: str, sha256: str) -> Document | None: ...

    def find_sha256_across_series(self, sha256: str) -> list[Document]: ...

    def record_parse_result(
        self, document_id: str, failed_page_count: int, pages_without_text_layer: int
    ) -> None: ...

    def mark_indexing(self, document_id: str, owner_id: str, lease_seconds: int) -> None: ...

    def heartbeat(self, document_id: str, owner_id: str, lease_seconds: int) -> None: ...

    def mark_ready(self, document_id: str, chunk_count: int) -> None: ...

    def mark_index_failed(self, document_id: str, error_message: str) -> None: ...

    def activate(self, document_id: str) -> None: ...

    def active_document_ids(self) -> list[str]: ...

    def recover_stale_indexing(self, stale_after_seconds: int) -> list[str]: ...


class PdfParser(Protocol):
    """PDF에서 페이지 단위 텍스트를 추출한다.

    문서를 열지 못하면 ParsingError를 던지고, 반쯤 처리된 결과를 돌려주지 않는다.
    개별 페이지의 추출 오류는 예외가 아니라 ParsedDocument.failed_pages로 남긴다.
    페이지 하나 때문에 문서 전체를 잃는 것보다 실패 위치를 식별한 채
    계속 가는 쪽이 FR-002의 요구와 맞다.

    parser_version은 문서 재색인 판단에 쓰인다. 파서가 바뀌면 같은 PDF에서
    다른 텍스트가 나올 수 있으므로 Document.parser_version과 대조한다(NFR-004).
    """

    @property
    def parser_version(self) -> str: ...

    def parse(self, pdf_path: Path) -> ParsedDocument: ...


class Chunker(Protocol):
    """파싱된 문서를 검색 단위로 나눈다.

    같은 입력에서 항상 같은 chunk_id·본문·페이지 범위가 나와야 한다(01 §12).
    결정론이 깨지면 재실행할 때마다 Qdrant에 다른 ID로 중복 벡터가 쌓인다.

    빈 문서는 오류가 아니라 빈 목록이다. 색인할 것이 없다는 사실은
    Ingestion 단계에서 문서 상태로 다루지, 예외로 다루지 않는다.
    """

    @property
    def chunk_config_version(self) -> str: ...

    def chunk(
        self,
        parsed_document: ParsedDocument,
        document_id: str,
        document_version: int,
    ) -> list[Chunk]: ...


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

    upsert는 chunk_id로 결정론적 ID를 만들어 재실행해도 중복 벡터가 생기지 않아야 한다.
    chunk_id는 전역에서 유일해야 한다. 지금은 청커가 document_id를 접두어로 붙여
    그 조건이 성립한다.

    IndexRun의 값들은 청크마다 같지만 payload에 복제한다. 문서 계열이나 종류로
    검색을 좁히려면 벡터 저장소가 그 값을 알아야 하고, 모르면 검색 결과마다
    문서 장부를 다시 조회하게 된다.
    """

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        index_run: IndexRun,
    ) -> None: ...

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
