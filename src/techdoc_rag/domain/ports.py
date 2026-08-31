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
    이전 버전으로 되돌리는 것도 새 버전을 등록해서 한다. INACTIVE에서 돌아오는
    전이는 없다.

    색인 결과를 확정하는 메서드는 소유자를 함께 받는다. 소유권을 잃은 프로세스가
    결과를 쓰면 복구가 이미 강등한 문서를 되살리게 된다.
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

    def mark_ready(
        self,
        document_id: str,
        owner_id: str,
        chunk_count: int,
        *,
        parser_version: str,
        chunk_config_version: str,
        embedding_model: str,
        embedding_version: str,
    ) -> None: ...

    def mark_index_failed(
        self, document_id: str, owner_id: str, error_message: str
    ) -> None: ...

    def activate(self, document_id: str) -> None: ...

    def active_document_ids(self) -> list[str]: ...

    def recover_abandoned_indexing(self, owner_id: str | None = None) -> list[str]: ...


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

    **한 번의 호출은 한 배치다.** 배치 크기는 생성자로 주입받고, 나누는 것은
    호출자가 한다. 어댑터가 안에서 나누면 호출자는 제어를 언제 돌려받을지 모른다.
    실물은 문서 하나가 평균 409청크이고 처리량이 0.64 chunk/s라 한 문서를 통째로
    넘기면 약 640초 동안 돌아오지 않는다. 그동안 호출자는 색인 리스를 갱신할 수
    없고, 복구가 살아 있는 색인을 강등한다. 리스 기본값은 300초다.

    **입력과 출력의 순서가 같아야 한다.** 호출자가 청크와 벡터를 위치로 짝지어
    저장한다. 배치 처리에서 순서가 어긋나면 오류 없이 청크와 벡터만 뒤바뀐다.

    **max_input_tokens를 넘는 입력은 초과분이 오류 없이 버려진다.** 모델이 실제로
    받는 길이를 알려야 색인 서비스가 청크 길이를 확인할 수 있다. 이 값은 모델
    태그나 로드된 컨텍스트 길이가 아니다 — Ollama는 컨텍스트가 8192여도 배치
    크기(`num_batch`, 기본 2048)에서 자른다. 구현체가 그 값을 직접 지정하고
    지정한 값을 돌려준다.

    **HTTP로 접속하는 구현체는 커넥션을 재사용하고 127.0.0.1로 접속한다.**
    요청마다 새로 열면 Windows에서 TIME_WAIT가 쌓여 동적 포트가 고갈된다.
    `localhost`는 IPv6로 먼저 해석되는데 Ollama가 IPv4에만 바인딩되어 있어
    폴백이 일어난다. 2026-08-30 실측에서 2,162ms 대 86ms였고, 2,454청크면
    약 80분 차이다.

    **model_name과 dimension은 실제 로드된 모델의 값이어야 한다.** 색인과 질의가
    다른 모델을 쓰면 차원이 같은 경우 오류 없이 검색 품질만 떨어진다(05 CR-06).
    embedding_version은 모델이 같아도 처리 방식이 바뀐 것을 구분한다(NFR-004).

    실패는 IndexingError로 올린다. 임베딩은 색인 경로의 일부다.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def embedding_version(self) -> str: ...

    @property
    def max_input_tokens(self) -> int: ...

    @property
    def batch_size(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """청크 벡터를 저장하고 검색한다.

    upsert는 chunk_id로 결정론적 ID를 만들어 재실행해도 중복 벡터가 생기지 않아야 한다.
    chunk_id는 전역에서 유일해야 한다. 지금은 청커가 document_id를 접두어로 붙여
    그 조건이 성립한다.

    IndexRun의 값들은 청크마다 같지만 payload에 복제한다. 문서 계열이나 종류로
    검색을 좁히려면 벡터 저장소가 그 값을 알아야 하고, 모르면 검색 결과마다
    SQLite를 다시 조회하게 된다.
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

    생성 지연이 길어 스트리밍을 기본으로 둔다. 4B 모델이 CPU에서 초당 8토큰이라
    300토큰 답변에 37초가 걸린다.

    **HTTP 커넥션을 재사용하고 127.0.0.1로 접속한다.** 요청마다 새로 열면
    Windows에서 TIME_WAIT가 쌓여 동적 포트가 고갈되고, 평가 스크립트가 문항을
    연속 처리할 때 실패한다.

    **동시 생성 수를 제한한다.** 설정값으로 받고 그 수를 넘는 요청은 대기시킨다.
    커넥션 하나를 재사용하는 이상 요청 둘이 동시에 오면 응답이 섞이므로,
    재사용을 택한 이상 직렬화는 피할 수 없다. 기본값 1은 CPU 환경을 전제한 것이고
    측정으로 확인하지 않았다. GPU를 확보하거나 추론 서버를 분리하면 올린다(DP-51).

    **타임아웃을 건다.** 서버가 응답을 멈추면 무한정 기다리게 된다.
    대기와 생성 각각에 상한을 두고, 넘으면 GenerationError로 끊는다.

    **첫 토큰 지연과 생성 지연을 나눠 잰다.** 동시 생성 제한 때문에 호출자가 재는
    "첫 토큰까지"에는 대기 시간이 섞인다. 구현체가 실제 값을 남겨야 둘을
    구분할 수 있다(NFR-005, 01 §17.8).

    **이터레이터를 닫으면 생성을 멈춘다.** 사용자가 화면을 닫으면 서비스가
    이터레이터를 버리는데, 그때 서버 쪽 생성이 계속 돌면 CPU를 점유한다.
    제너레이터 함수로 구현해 GeneratorExit를 받는다. 호출자는 다 쓰지 않을
    이터레이터를 명시적으로 닫는다 — GC에 맡기면 멈추는 시점이 불확정이다.

    **취소한 커넥션은 버리고 새로 연다.** 스트림을 중간에 닫으면 읽지 않은
    바이트가 남아, 그 커넥션을 재사용하면 다음 응답의 상태줄을 잘못 읽는다.
    커넥션 재사용과 취소는 이 처리 없이는 같이 성립하지 않는다.

    빈 응답은 실패가 아니다. 근거가 부족하다는 판단은 상위에서 한다.
    """

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]: ...
