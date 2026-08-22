"""등록된 기술문서와 그 생명주기."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    """문서 상태.

    INDEXING 중 프로세스가 죽으면 상태가 영구히 고착되므로,
    저장소 계층에서 heartbeat 기반 복구가 필요하다.
    """

    NEW = "NEW"
    INDEXING = "INDEXING"
    READY = "READY"
    INDEX_FAILED = "INDEX_FAILED"
    INACTIVE = "INACTIVE"
    DELETING = "DELETING"
    DELETED = "DELETED"
    DELETE_FAILED = "DELETE_FAILED"


@dataclass(frozen=True, slots=True)
class Document:
    """문서 한 버전.

    같은 문서의 여러 버전은 logical_document_id를 공유하고 document_id로 구분한다.
    검색 대상은 is_active인 버전으로 한정한다.

    원본 파일 경로를 필드로 두지 않는 것은 의도적이다. DB에 저장된 경로를 그대로 열면
    경로 조작에 노출되므로, 파일 위치는 document_id로부터 결정론적으로 재구성한다.
    """

    document_id: str
    logical_document_id: str
    document_version: int
    original_filename: str
    sha256: str
    mime_type: str
    file_size_bytes: int
    page_count: int
    document_type: str
    status: DocumentStatus
    is_active: bool
    created_at: datetime

    # 재현성을 위해 어떤 설정으로 색인했는지 함께 보관한다.
    parser_version: str | None = None
    chunk_config_version: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None

    indexed_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def is_searchable(self) -> bool:
        return self.is_active and self.status is DocumentStatus.READY
