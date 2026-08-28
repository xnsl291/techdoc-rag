"""SQLite 문서 저장소.

문서 생명주기의 정본. Qdrant는 검색 인덱스일 뿐이고 어떤 문서가 유효한지는
여기가 결정한다(D-004). 검색 시 active 문서 ID 목록을 여기서 읽어
Vector Store 필터로 넘긴다.

정합성은 코드가 아니라 DB 제약으로 강제한다.

- active 1개 불변식: partial unique index (SQLite 3.8.0+)
- 버전·해시 중복: UNIQUE 제약
- 상태 오타: CHECK 제약

접속 설정(WAL, busy_timeout)은 Windows에서 SQLite 락킹이 깨지는 문제와
동시 접근 대비다. 파일은 bind mount가 아니라 로컬 경로에 둬야 한다.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import MetadataStoreError

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in DocumentStatus)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    document_id          TEXT PRIMARY KEY,
    logical_document_id  TEXT NOT NULL,
    document_version     INTEGER NOT NULL,
    original_filename    TEXT NOT NULL,
    sha256               TEXT NOT NULL,
    mime_type            TEXT NOT NULL,
    file_size_bytes      INTEGER NOT NULL,
    page_count           INTEGER NOT NULL,
    document_type        TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ({_STATUS_VALUES})),
    is_active            INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    created_at           TEXT NOT NULL,
    parser_version       TEXT,
    chunk_config_version TEXT,
    embedding_model      TEXT,
    embedding_version    TEXT,
    indexed_at           TEXT,
    deleted_at           TEXT,
    -- 운영 컬럼. INDEXING 중 죽은 프로세스를 식별하고 복구하는 데 쓴다(CR-02).
    indexing_started_at  TEXT,
    heartbeat_at         TEXT,
    attempt_count        INTEGER NOT NULL DEFAULT 0,
    last_error           TEXT,
    chunk_count          INTEGER,
    UNIQUE (logical_document_id, document_version),
    UNIQUE (logical_document_id, sha256)
);

-- 불변식: 같은 논리 문서에서 active 버전은 1개.
-- 코드에 버그가 있어도 두 번째 active INSERT/UPDATE는 여기서 거부된다(CR-03).
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_active
    ON documents (logical_document_id) WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SqliteDocumentRepository:
    """DocumentRepository의 SQLite 구현."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        # WAL: 읽기와 쓰기가 서로를 덜 막음. busy_timeout: 잠금 경합 시 즉시 실패하지 않음.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._connection.close()

    def initialize(self) -> None:
        try:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        except sqlite3.Error as exc:
            raise MetadataStoreError(f"스키마 생성 실패: {exc}") from exc

    def register(self, document: Document) -> None:
        try:
            self._connection.execute(
                """
                INSERT INTO documents (
                    document_id, logical_document_id, document_version,
                    original_filename, sha256, mime_type, file_size_bytes,
                    page_count, document_type, status, is_active, created_at,
                    parser_version, chunk_config_version, embedding_model,
                    embedding_version, indexed_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.logical_document_id,
                    document.document_version,
                    document.original_filename,
                    document.sha256,
                    document.mime_type,
                    document.file_size_bytes,
                    document.page_count,
                    document.document_type,
                    document.status.value,
                    int(document.is_active),
                    document.created_at.isoformat(),
                    document.parser_version,
                    document.chunk_config_version,
                    document.embedding_model,
                    document.embedding_version,
                    document.indexed_at.isoformat() if document.indexed_at else None,
                    document.deleted_at.isoformat() if document.deleted_at else None,
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise MetadataStoreError(
                f"문서 등록 거부: {document.document_id} "
                f"(중복 버전·해시 또는 제약 위반: {exc})"
            ) from exc
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"문서 등록 실패: {exc}") from exc

    def get(self, document_id: str) -> Document | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _to_document(row) if row else None

    def find_by_sha256(self, logical_document_id: str, sha256: str) -> Document | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE logical_document_id = ? AND sha256 = ?",
            (logical_document_id, sha256),
        ).fetchone()
        return _to_document(row) if row else None

    def mark_indexing(self, document_id: str) -> None:
        now = _now_iso()
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET status = ?, indexing_started_at = ?, heartbeat_at = ?,
                   attempt_count = attempt_count + 1, last_error = NULL
             WHERE document_id = ?
            """,
            (DocumentStatus.INDEXING.value, now, now, document_id),
        )

    def heartbeat(self, document_id: str) -> None:
        self._update_status(
            document_id,
            "UPDATE documents SET heartbeat_at = ? WHERE document_id = ? AND status = ?",
            (_now_iso(), document_id, DocumentStatus.INDEXING.value),
        )

    def mark_ready(self, document_id: str, chunk_count: int) -> None:
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET status = ?, indexed_at = ?, chunk_count = ?
             WHERE document_id = ?
            """,
            (DocumentStatus.READY.value, _now_iso(), chunk_count, document_id),
        )

    def mark_index_failed(self, document_id: str, error_message: str) -> None:
        self._update_status(
            document_id,
            "UPDATE documents SET status = ?, last_error = ? WHERE document_id = ?",
            (DocumentStatus.INDEX_FAILED.value, error_message, document_id),
        )

    def activate(self, document_id: str) -> None:
        """READY인 버전을 active로 올리고 같은 계열의 기존 active를 내린다.

        한 트랜잭션이다. 중간에 죽으면 전부 되돌아가 기존 active가 유지된다
        (01 §17.6: 실패 시 기존 서비스 검색 상태 유지). partial unique index
        때문에 트랜잭션 안에서 비활성화를 먼저 한다.
        """
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT logical_document_id, status FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise MetadataStoreError(f"활성 전환 실패: 문서 없음 ({document_id})")
            if row["status"] != DocumentStatus.READY.value:
                raise MetadataStoreError(
                    f"활성 전환 거부: {document_id}는 READY가 아님 (현재 {row['status']})"
                )
            self._connection.execute(
                """
                UPDATE documents SET is_active = 0, status = ?
                 WHERE logical_document_id = ? AND is_active = 1 AND document_id != ?
                """,
                (DocumentStatus.INACTIVE.value, row["logical_document_id"], document_id),
            )
            self._connection.execute(
                "UPDATE documents SET is_active = 1 WHERE document_id = ?",
                (document_id,),
            )
            self._connection.commit()
        except MetadataStoreError:
            self._connection.rollback()
            raise
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"활성 전환 실패: {document_id} ({exc})") from exc

    def active_document_ids(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT document_id FROM documents WHERE is_active = 1 AND status = ?",
            (DocumentStatus.READY.value,),
        ).fetchall()
        return [row["document_id"] for row in rows]

    def recover_stale_indexing(self, stale_after_seconds: int) -> list[str]:
        """heartbeat이 끊긴 INDEXING 문서를 INDEX_FAILED로 강등한다.

        INDEXING에서 나가는 전이는 원래 프로세스가 살아 있어야 기록되므로,
        죽은 프로세스가 남긴 문서는 이 복구가 없으면 영구 고착된다(CR-02).
        기동 시 한 번 호출한다.
        """
        cutoff = datetime.now(UTC).timestamp() - stale_after_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff, UTC).isoformat()
        try:
            rows = self._connection.execute(
                """
                UPDATE documents
                   SET status = ?, last_error = 'stale heartbeat: 색인 중 프로세스 중단으로 판정'
                 WHERE status = ? AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                RETURNING document_id
                """,
                (DocumentStatus.INDEX_FAILED.value, DocumentStatus.INDEXING.value, cutoff_iso),
            ).fetchall()
            self._connection.commit()
            return [row["document_id"] for row in rows]
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"stale INDEXING 복구 실패: {exc}") from exc

    def _update_status(self, document_id: str, sql: str, parameters: tuple) -> None:
        try:
            cursor = self._connection.execute(sql, parameters)
            if cursor.rowcount == 0:
                self._connection.rollback()
                raise MetadataStoreError(f"상태 갱신 실패: 대상 없음 ({document_id})")
            self._connection.commit()
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"상태 갱신 실패: {document_id} ({exc})") from exc


def _to_document(row: sqlite3.Row) -> Document:
    def _parse_datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    created_at = _parse_datetime(row["created_at"])
    assert created_at is not None  # NOT NULL 컬럼
    return Document(
        document_id=row["document_id"],
        logical_document_id=row["logical_document_id"],
        document_version=row["document_version"],
        original_filename=row["original_filename"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        file_size_bytes=row["file_size_bytes"],
        page_count=row["page_count"],
        document_type=row["document_type"],
        status=DocumentStatus(row["status"]),
        is_active=bool(row["is_active"]),
        created_at=created_at,
        parser_version=row["parser_version"],
        chunk_config_version=row["chunk_config_version"],
        embedding_model=row["embedding_model"],
        embedding_version=row["embedding_version"],
        indexed_at=_parse_datetime(row["indexed_at"]),
        deleted_at=_parse_datetime(row["deleted_at"]),
    )
