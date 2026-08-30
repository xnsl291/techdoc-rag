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

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import MetadataStoreError
from techdoc_rag.domain.indexing import validate_logical_document_id

_logger = logging.getLogger(__name__)

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
    -- 누가 이 문서를 색인 중인지. 없으면 두 번째 프로세스가 살아 있는 색인을
    -- 죽은 것으로 보고 강등한다.
    owner_id             TEXT,
    lease_until          TEXT,
    -- 파싱 결과. 색인이 끝나면 메모리에서 사라지므로 여기 남긴다(FR-002).
    failed_page_count    INTEGER,
    pages_without_text_layer INTEGER,
    UNIQUE (logical_document_id, document_version),
    UNIQUE (logical_document_id, sha256)
);

-- 불변식: 같은 논리 문서에서 active 버전은 1개.
-- 코드에 버그가 있어도 두 번째 active INSERT/UPDATE는 여기서 거부된다(CR-03).
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_active
    ON documents (logical_document_id) WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status);
"""


# 스키마가 최신인지 확인하는 데 쓴다. 새로 추가한 컬럼만 담는다.
_EXPECTED_COLUMNS = {
    "owner_id",
    "lease_until",
    "failed_page_count",
    "pages_without_text_layer",
}


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
        """스키마를 만들고, 이미 있으면 기대하는 컬럼이 다 있는지 확인한다.

        CREATE TABLE IF NOT EXISTS는 기존 테이블에 컬럼을 더하지 않는다.
        옛 스키마로 만든 DB에 그대로 붙으면 첫 상태 갱신에서 no such column이
        SQL 오류로 감싸져 나와 원인을 알 수 없다. 여기서 끊는다.
        """
        try:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        except sqlite3.Error as exc:
            raise MetadataStoreError(f"스키마 생성 실패: {exc}") from exc

        existing = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        missing = _EXPECTED_COLUMNS - existing
        if missing:
            raise MetadataStoreError(
                f"documents 테이블에 컬럼이 없음: {sorted(missing)}. "
                f"옛 스키마로 만든 DB임. 개발 중이면 파일을 지우고 다시 만들 것"
            )

    def register(self, document: Document) -> None:
        """문서를 등록한다.

        계열이 다른데 같은 해시가 이미 있으면 경고만 남기고 등록은 진행한다.
        계열 ID 오타로 같은 파일이 두 번 들어오면 둘 다 활성이 되어 같은 근거가
        두 번 검색된다. 다만 같은 PDF가 두 제품에 공용으로 쓰이는 경우도 있어
        막지는 않는다.
        """
        validate_logical_document_id(document.logical_document_id)
        duplicates = [
            existing
            for existing in self.find_sha256_across_series(document.sha256)
            if existing.logical_document_id != document.logical_document_id
        ]
        if duplicates:
            _logger.warning(
                "같은 해시가 다른 계열에 이미 있음: %s (해시 %s, 기존 계열 %s). "
                "계열 ID 오타인지 확인할 것",
                document.document_id,
                document.sha256[:12],
                sorted({existing.logical_document_id for existing in duplicates}),
            )
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

    def find_sha256_across_series(self, sha256: str) -> list[Document]:
        """계열을 가리지 않고 같은 해시를 가진 문서를 찾는다.

        중복 등록은 UNIQUE (logical_document_id, sha256)이 계열 안에서만 막는다.
        계열 ID에 오타가 나면 같은 파일이 다른 계열로 들어가고, 둘 다 활성이 되어
        같은 근거가 두 번 검색된다.

        여기서 막지 않고 알리기만 한다. 같은 PDF가 두 제품에 공용으로 쓰이는
        정당한 경우가 있어서, 등록을 거부할지는 색인 서비스가 정한다.
        """
        rows = self._connection.execute(
            "SELECT * FROM documents WHERE sha256 = ? ORDER BY logical_document_id",
            (sha256,),
        ).fetchall()
        return [_to_document(row) for row in rows]

    def record_parse_result(
        self, document_id: str, failed_page_count: int, pages_without_text_layer: int
    ) -> None:
        """파싱 결과를 남긴다. 이 값들은 색인이 끝나면 메모리에서 사라진다(FR-002)."""
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET failed_page_count = ?, pages_without_text_layer = ?
             WHERE document_id = ?
            """,
            (failed_page_count, pages_without_text_layer, document_id),
        )

    def mark_indexing(self, document_id: str, owner_id: str, lease_seconds: int) -> None:
        """색인 시작을 기록하고 소유권을 잡는다.

        NEW와 INDEX_FAILED에서만 들어올 수 있다. READY 문서를 제자리에서 다시
        색인하면 is_active가 1인 채로 검색에서 빠지므로 막는다(DP-50).
        재색인은 항상 새 버전을 등록해서 한다.

        owner_id는 이 색인을 잡은 프로세스를 가리킨다. 없으면 다른 프로세스가
        살아 있는 색인을 죽은 것으로 보고 강등시킬 수 있다.
        """
        now = datetime.now(UTC)
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET status = ?, indexing_started_at = ?, heartbeat_at = ?,
                   owner_id = ?, lease_until = ?,
                   attempt_count = attempt_count + 1, last_error = NULL
             WHERE document_id = ? AND status IN (?, ?)
            """,
            (
                DocumentStatus.INDEXING.value,
                now.isoformat(),
                now.isoformat(),
                owner_id,
                (now + timedelta(seconds=lease_seconds)).isoformat(),
                document_id,
                DocumentStatus.NEW.value,
                DocumentStatus.INDEX_FAILED.value,
            ),
            failure_hint="NEW 또는 INDEX_FAILED 상태에서만 색인을 시작할 수 있음. "
            "이미 색인된 문서를 다시 색인하려면 새 버전으로 등록할 것 (DP-50)",
        )

    def heartbeat(self, document_id: str, owner_id: str, lease_seconds: int) -> None:
        """살아 있음을 알리고 리스를 연장한다.

        소유자가 아니면 갱신되지 않는다. 이미 다른 프로세스에 넘어간 문서를
        계속 갱신하면 두 프로세스가 같은 문서를 색인하게 된다.
        """
        now = datetime.now(UTC)
        self._update_status(
            document_id,
            """
            UPDATE documents SET heartbeat_at = ?, lease_until = ?
             WHERE document_id = ? AND status = ? AND owner_id = ?
            """,
            (
                now.isoformat(),
                (now + timedelta(seconds=lease_seconds)).isoformat(),
                document_id,
                DocumentStatus.INDEXING.value,
                owner_id,
            ),
            failure_hint="색인 중이 아니거나 소유자가 아님",
        )

    def mark_ready(self, document_id: str, owner_id: str, chunk_count: int) -> None:
        """색인 완료를 확정한다. 소유권을 잃었으면 쓸 수 없다.

        조건이 없으면 복구가 이미 강등한 문서를 원래 프로세스가 READY로 되돌린다.
        강등 기록과 실제 상태가 어긋나고, 소유권 없는 프로세스의 결과가 확정된다.
        """
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET status = ?, indexed_at = ?, chunk_count = ?,
                   owner_id = NULL, lease_until = NULL
             WHERE document_id = ? AND status = ? AND owner_id = ?
            """,
            (
                DocumentStatus.READY.value,
                _now_iso(),
                chunk_count,
                document_id,
                DocumentStatus.INDEXING.value,
                owner_id,
            ),
        )

    def mark_index_failed(self, document_id: str, owner_id: str, error_message: str) -> None:
        self._update_status(
            document_id,
            """
            UPDATE documents
               SET status = ?, last_error = ?, owner_id = NULL, lease_until = NULL
             WHERE document_id = ? AND status = ? AND owner_id = ?
            """,
            (
                DocumentStatus.INDEX_FAILED.value,
                error_message,
                document_id,
                DocumentStatus.INDEXING.value,
                owner_id,
            ),
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

    def recover_abandoned_indexing(self, owner_id: str | None = None) -> list[str]:
        """버려진 INDEXING 문서를 INDEX_FAILED로 강등한다.

        INDEXING에서 나가는 전이는 원래 프로세스가 살아 있어야 기록되므로,
        죽은 프로세스가 남긴 문서는 이 복구가 없으면 영구 고착된다(CR-02).
        기동 시 한 번, 그리고 주기적으로 호출한다.

        두 가지를 회수한다.

        - **리스가 만료된 것.** 살아 있는 프로세스는 heartbeat으로 리스를 계속
          연장하므로, 만료됐다는 것은 그 프로세스가 멈췄다는 뜻이다
        - **owner_id를 주면 그 소유자의 것.** 리스가 남아 있어도 회수한다.
          같은 owner_id로 다시 올라온 프로세스는 정의상 이전 자신이 죽었다는 뜻이다.
          이게 없으면 리스(수 분)보다 빨리 재기동했을 때 그 문서가 영구히 갇힌다

        후자가 성립하려면 owner_id가 재기동에도 유지되는 값이어야 한다.
        호스트명과 워커 번호처럼 프로세스가 아니라 자리를 가리키는 값을 쓴다.
        """
        conditions = ["lease_until IS NULL", "lease_until < ?"]
        parameters: list[str] = [_now_iso()]
        if owner_id is not None:
            conditions.append("owner_id = ?")
            parameters.append(owner_id)

        try:
            rows = self._connection.execute(
                f"""
                UPDATE documents
                   SET status = ?,
                       last_error = '색인 중 프로세스가 중단된 것으로 판정',
                       owner_id = NULL, lease_until = NULL
                 WHERE status = ? AND ({" OR ".join(conditions)})
                RETURNING document_id
                """,
                (DocumentStatus.INDEX_FAILED.value, DocumentStatus.INDEXING.value, *parameters),
            ).fetchall()
            self._connection.commit()
            return [row["document_id"] for row in rows]
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"버려진 색인 복구 실패: {exc}") from exc

    def _update_status(
        self, document_id: str, sql: str, parameters: tuple, failure_hint: str = ""
    ) -> None:
        try:
            cursor = self._connection.execute(sql, parameters)
            if cursor.rowcount == 0:
                self._connection.rollback()
                raise MetadataStoreError(self._explain_no_match(document_id, failure_hint))
            self._connection.commit()
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise MetadataStoreError(f"상태 갱신 실패: {document_id} ({exc})") from exc

    def _explain_no_match(self, document_id: str, failure_hint: str) -> str:
        """갱신 대상이 없는 이유를 찾아 메시지로 만든다.

        rowcount가 0인 원인은 여럿이다. 문서가 없거나, 상태가 안 맞거나,
        소유자가 다르다. 고정 문구를 쓰면 로그만 보고 원인을 반대로 짚게 된다.
        오류 경로라 조회를 한 번 더 하는 비용은 문제가 되지 않는다.
        """
        row = self._connection.execute(
            "SELECT status, owner_id FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return f"상태 갱신 실패: {document_id} — 문서 없음"
        detail = f"현재 상태 {row['status']}, 소유자 {row['owner_id'] or '없음'}"
        hint = f". {failure_hint}" if failure_hint else ""
        return f"상태 갱신 실패: {document_id} — 조건 불일치 ({detail}){hint}"


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
        failed_page_count=row["failed_page_count"],
        pages_without_text_layer=row["pages_without_text_layer"],
    )
