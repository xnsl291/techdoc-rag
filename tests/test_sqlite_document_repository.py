"""SqliteDocumentRepository 테스트.

핵심 검증 대상은 CRUD가 아니라 정합성이다 — active 1개 불변식이 DB 제약으로
막히는지, 활성 전환이 원자적인지, 죽은 INDEXING이 복구되는지.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from techdoc_rag.adapters.sqlite_document_repository import SqliteDocumentRepository
from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import MetadataStoreError


def _document(
    document_id: str = "doc-1",
    logical_document_id: str = "manual-g100",
    document_version: int = 1,
    sha256: str = "a" * 64,
) -> Document:
    return Document(
        document_id=document_id,
        logical_document_id=logical_document_id,
        document_version=document_version,
        original_filename="manual.pdf",
        sha256=sha256,
        mime_type="application/pdf",
        file_size_bytes=1024,
        page_count=10,
        document_type="manual",
        status=DocumentStatus.NEW,
        is_active=False,
        created_at=datetime.now(UTC),
    )


@pytest.fixture()
def repository(tmp_path: Path) -> SqliteDocumentRepository:
    repo = SqliteDocumentRepository(tmp_path / "metadata.db")
    repo.initialize()
    yield repo
    repo.close()


def test_등록과_조회가_모든_필드를_보존한다(repository: SqliteDocumentRepository) -> None:
    document = _document()
    repository.register(document)

    loaded = repository.get("doc-1")

    assert loaded == document


def test_같은_계열에_같은_해시는_거부된다(repository: SqliteDocumentRepository) -> None:
    repository.register(_document(document_id="doc-1", sha256="f" * 64))

    with pytest.raises(MetadataStoreError):
        repository.register(
            _document(document_id="doc-2", document_version=2, sha256="f" * 64)
        )


def test_다른_계열이면_같은_해시도_허용된다(repository: SqliteDocumentRepository) -> None:
    repository.register(_document(document_id="doc-1", sha256="f" * 64))
    repository.register(
        _document(document_id="doc-2", logical_document_id="manual-s100", sha256="f" * 64)
    )

    assert repository.find_by_sha256("manual-s100", "f" * 64) is not None


def test_색인_생명주기와_활성_전환(repository: SqliteDocumentRepository) -> None:
    # v1을 READY+active로 만든다.
    repository.register(_document(document_id="v1", document_version=1, sha256="a" * 64))
    repository.mark_indexing("v1")
    repository.mark_ready("v1", chunk_count=100)
    repository.activate("v1")
    assert repository.active_document_ids() == ["v1"]

    # v2가 READY가 된 뒤 활성 전환하면 v1이 내려간다.
    repository.register(_document(document_id="v2", document_version=2, sha256="b" * 64))
    repository.mark_indexing("v2")
    repository.mark_ready("v2", chunk_count=105)
    repository.activate("v2")

    assert repository.active_document_ids() == ["v2"]
    v1 = repository.get("v1")
    assert v1 is not None
    assert v1.status is DocumentStatus.INACTIVE
    assert v1.is_active is False


def test_active_2개는_DB_제약이_거부한다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """저장소 코드를 우회해 직접 SQL로 시도해도 막혀야 한다.

    이것이 이 이슈의 핵심이다. 코드로 지키는 규칙은 버그 하나에 뚫리지만
    partial unique index는 경로와 무관하게 거부한다(CR-03).
    """
    repository.register(_document(document_id="v1", document_version=1, sha256="a" * 64))
    repository.mark_indexing("v1")
    repository.mark_ready("v1", chunk_count=1)
    repository.activate("v1")
    repository.register(_document(document_id="v2", document_version=2, sha256="b" * 64))

    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            raw_connection.execute("UPDATE documents SET is_active = 1 WHERE document_id = 'v2'")
    finally:
        raw_connection.close()


def test_READY가_아니면_활성_전환이_거부되고_기존_active가_유지된다(
    repository: SqliteDocumentRepository,
) -> None:
    repository.register(_document(document_id="v1", document_version=1, sha256="a" * 64))
    repository.mark_indexing("v1")
    repository.mark_ready("v1", chunk_count=1)
    repository.activate("v1")
    # v2는 INDEXING 상태에서 활성 전환을 시도한다.
    repository.register(_document(document_id="v2", document_version=2, sha256="b" * 64))
    repository.mark_indexing("v2")

    with pytest.raises(MetadataStoreError):
        repository.activate("v2")

    # 실패한 전환이 기존 검색 상태를 건드리지 않았다(01 §17.6).
    assert repository.active_document_ids() == ["v1"]


def test_heartbeat이_끊긴_INDEXING은_복구에서_강등된다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    repository.register(_document(document_id="stale", sha256="a" * 64))
    repository.mark_indexing("stale")
    repository.register(
        _document(
            document_id="alive",
            logical_document_id="manual-s100",
            sha256="b" * 64,
        )
    )
    repository.mark_indexing("alive")

    # stale의 heartbeat을 과거로 되돌린다.
    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    raw_connection.execute(
        "UPDATE documents SET heartbeat_at = '2020-01-01T00:00:00+00:00'"
        " WHERE document_id = 'stale'"
    )
    raw_connection.commit()
    raw_connection.close()

    demoted = repository.recover_stale_indexing(stale_after_seconds=600)

    assert demoted == ["stale"]
    stale = repository.get("stale")
    assert stale is not None
    assert stale.status is DocumentStatus.INDEX_FAILED
    alive = repository.get("alive")
    assert alive is not None
    assert alive.status is DocumentStatus.INDEXING


def test_없는_문서의_상태_갱신은_실패한다(repository: SqliteDocumentRepository) -> None:
    with pytest.raises(MetadataStoreError):
        repository.mark_indexing("ghost")


def test_WAL_모드로_동작한다(repository: SqliteDocumentRepository, tmp_path: Path) -> None:
    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    mode = raw_connection.execute("PRAGMA journal_mode").fetchone()[0]
    raw_connection.close()

    assert mode == "wal"


def test_재색인_시도마다_attempt_count가_올라간다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    repository.register(_document(document_id="doc-1"))
    repository.mark_indexing("doc-1")
    repository.mark_index_failed("doc-1", "첫 실패")
    repository.mark_indexing("doc-1")

    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    count = raw_connection.execute(
        "SELECT attempt_count FROM documents WHERE document_id = 'doc-1'"
    ).fetchone()[0]
    raw_connection.close()

    assert count == 2
