"""색인 서비스 — 등록부터 READY까지 한 문서의 색인을 끝까지 책임진다 (#17).

순서가 계약이다(01 §17.6).

등록 → INDEXING(소유권+리스) → 파싱 → 파싱 결과 기록 → 청킹
→ [임베딩 → 저장 → heartbeat]를 배치마다 반복 → 고아 벡터 정리 → READY

heartbeat를 배치 사이에 넣는 이유: 임베딩이 색인에서 가장 긴 단계라
(개발기 0.64 chunk/s에서 문서 하나 약 10분) 문서 단위로만 갱신하면
리스(기본 300초)가 그 안에 만료되어 복구가 살아 있는 색인을 강등한다.

활성 전환(activate)은 여기서 하지 않는다. 색인 완료와 검색 노출은
별개의 결정이고, 운영자가 READY 상태를 확인한 뒤 전환한다(01 §17.6).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import IndexingError, ParsingError
from techdoc_rag.domain.indexing import IndexRun, new_index_run_id
from techdoc_rag.domain.ports import (
    Chunker,
    DocumentRepository,
    EmbeddingModel,
    PdfParser,
    VectorStore,
)

# 같은 해시가 이미 있어도 색인을 다시 시도해야 하는 상태.
# mark_indexing이 받아 주는 상태와 같다 — 저장소의 재시도 규칙을 그대로 따른다.
_RETRYABLE = (DocumentStatus.NEW, DocumentStatus.INDEX_FAILED)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """색인 한 번의 결과.

    duplicate_of가 있으면 같은 계열에 같은 해시가 이미 있어 색인하지 않은 것이다.
    상태 세부는 저장소가 정본이므로 여기 복제하지 않는다.
    """

    document_id: str
    chunk_count: int
    duplicate_of: str | None = None


class IngestionService:
    def __init__(
        self,
        parser: PdfParser,
        chunker: Chunker,
        repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        owner_id: str,
        lease_seconds: int,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._repository = repository
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._owner_id = owner_id
        self._lease_seconds = lease_seconds

    def ingest(
        self,
        pdf_path: Path,
        logical_document_id: str,
        document_version: int,
        document_type: str,
    ) -> IngestionResult:
        """PDF 하나를 색인한다. 성공하면 READY, 실패하면 INDEX_FAILED로 남는다.

        document_id는 {logical_document_id}-v{version}으로 결정론적이다.
        같은 입력의 재실행이 새 ID를 만들지 않아야 벡터가 chunk_id로 덮어써진다.
        """
        sha256 = _file_sha256(pdf_path)
        existing = self._repository.find_by_sha256(logical_document_id, sha256)
        if existing is not None and existing.status not in _RETRYABLE:
            # 내용이 같은 파일을 같은 계열에 다시 넣은 것 — 새 버전이 아니다.
            # NEW/INDEX_FAILED는 여기서 걸러내면 안 된다. 걸러내면 실패한 문서가
            # "이미 처리됨"으로 반환되어 저장소가 열어 둔 재시도 경로가 막힌다(리뷰 B-1).
            return IngestionResult(
                document_id=existing.document_id,
                chunk_count=0,
                duplicate_of=existing.document_id,
            )

        if existing is not None:
            # 등록만 됐거나 색인이 실패한 문서 — 등록을 건너뛰고 그 자리에서 재개한다.
            document_id = existing.document_id
            document_version = existing.document_version
        else:
            document_id = f"{logical_document_id}-v{document_version}"
            self._repository.register(
                Document(
                    document_id=document_id,
                    logical_document_id=logical_document_id,
                    document_version=document_version,
                    original_filename=pdf_path.name,
                    sha256=sha256,
                    mime_type="application/pdf",
                    file_size_bytes=pdf_path.stat().st_size,
                    page_count=0,  # 파싱 전이라 모른다. record_parse_result가 채운다.
                    document_type=document_type,
                    status=DocumentStatus.NEW,
                    is_active=False,
                    created_at=datetime.now(UTC),
                )
            )
        self._repository.mark_indexing(
            document_id, owner_id=self._owner_id, lease_seconds=self._lease_seconds
        )

        try:
            chunk_count = self._parse_and_index(
                pdf_path, document_id, logical_document_id, document_type, document_version
            )
        except (ParsingError, IndexingError) as error:
            # 실패 원인을 상태에 남기고 예외는 그대로 올린다. 여기서 삼키면
            # 배치 색인 스크립트가 실패한 문서를 성공으로 센다.
            self._repository.mark_index_failed(
                document_id, owner_id=self._owner_id, error_message=str(error)
            )
            raise
        return IngestionResult(document_id=document_id, chunk_count=chunk_count)

    def _parse_and_index(
        self,
        pdf_path: Path,
        document_id: str,
        logical_document_id: str,
        document_type: str,
        document_version: int,
    ) -> int:
        parsed = self._parser.parse(pdf_path)
        self._repository.record_parse_result(
            document_id,
            page_count=parsed.page_count,
            failed_page_count=len(parsed.failed_pages),
            pages_without_text_layer=parsed.pages_without_text_layer,
        )

        chunks = self._chunker.chunk(parsed, document_id, document_version)
        if not chunks:
            raise IndexingError(f"색인할 텍스트 없음: {document_id} (스캔본이거나 빈 문서)")

        index_run = IndexRun(
            document_id=document_id,
            logical_document_id=logical_document_id,
            document_type=document_type,
            index_run_id=new_index_run_id(),
        )
        batch_size = self._embedding_model.batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embedding_model.embed_documents([chunk.text for chunk in batch])
            self._vector_store.upsert(batch, vectors, index_run)
            self._repository.heartbeat(
                document_id, owner_id=self._owner_id, lease_seconds=self._lease_seconds
            )

        # 저장이 다 끝난 뒤에 정리한다. 먼저 지우면 이번 실행이 실패했을 때
        # 이전 실행의 벡터까지 잃어 되돌릴 것이 없다.
        self._vector_store.delete_stale_runs(document_id, index_run.index_run_id)

        self._repository.mark_ready(
            document_id,
            owner_id=self._owner_id,
            chunk_count=len(chunks),
            parser_version=self._parser.parser_version,
            chunk_config_version=self._chunker.chunk_config_version,
            embedding_model=self._embedding_model.model_name,
            embedding_version=self._embedding_model.embedding_version,
        )
        return len(chunks)


def _file_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
