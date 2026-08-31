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
    logical_document_id: str = "ls-g100",
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
        _document(document_id="doc-2", logical_document_id="ls-s100", sha256="f" * 64)
    )

    assert repository.find_by_sha256("ls-s100", "f" * 64) is not None


def test_색인_생명주기와_활성_전환(repository: SqliteDocumentRepository) -> None:
    # v1을 READY+active로 만든다.
    repository.register(_document(document_id="v1", document_version=1, sha256="a" * 64))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v1", owner_id="worker-1", chunk_count=100)
    repository.activate("v1")
    assert repository.active_document_ids() == ["v1"]

    # v2가 READY가 된 뒤 활성 전환하면 v1이 내려간다.
    repository.register(_document(document_id="v2", document_version=2, sha256="b" * 64))
    repository.mark_indexing("v2", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v2", owner_id="worker-1", chunk_count=105)
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
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v1", owner_id="worker-1", chunk_count=1)
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
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v1", owner_id="worker-1", chunk_count=1)
    repository.activate("v1")
    # v2는 INDEXING 상태에서 활성 전환을 시도한다.
    repository.register(_document(document_id="v2", document_version=2, sha256="b" * 64))
    repository.mark_indexing("v2", owner_id="worker-1", lease_seconds=300)

    with pytest.raises(MetadataStoreError):
        repository.activate("v2")

    # 실패한 전환이 기존 검색 상태를 건드리지 않았다(01 §17.6).
    assert repository.active_document_ids() == ["v1"]


def test_리스가_만료된_INDEXING은_복구에서_강등된다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    repository.register(_document(document_id="stale", sha256="a" * 64))
    repository.mark_indexing("stale", owner_id="dead-worker", lease_seconds=300)
    repository.register(
        _document(
            document_id="alive",
            logical_document_id="ls-s100",
            sha256="b" * 64,
        )
    )
    repository.mark_indexing("alive", owner_id="live-worker", lease_seconds=300)

    # 프로세스가 죽어 리스가 만료된 상황을 만든다.
    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    raw_connection.execute(
        "UPDATE documents SET heartbeat_at = '2020-01-01T00:00:00+00:00',"
        " lease_until = '2020-01-01T00:05:00+00:00'"
        " WHERE document_id = 'stale'"
    )
    raw_connection.commit()
    raw_connection.close()

    demoted = repository.recover_abandoned_indexing()

    assert demoted == ["stale"]
    stale = repository.get("stale")
    assert stale is not None
    assert stale.status is DocumentStatus.INDEX_FAILED
    alive = repository.get("alive")
    assert alive is not None
    assert alive.status is DocumentStatus.INDEXING


def test_없는_문서의_상태_갱신은_실패한다(repository: SqliteDocumentRepository) -> None:
    with pytest.raises(MetadataStoreError):
        repository.mark_indexing("ghost", owner_id="worker-1", lease_seconds=300)


def test_WAL_모드로_동작한다(repository: SqliteDocumentRepository, tmp_path: Path) -> None:
    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    mode = raw_connection.execute("PRAGMA journal_mode").fetchone()[0]
    raw_connection.close()

    assert mode == "wal"


def test_재색인_시도마다_attempt_count가_올라간다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    repository.register(_document(document_id="doc-1"))
    repository.mark_indexing("doc-1", owner_id="worker-1", lease_seconds=300)
    repository.mark_index_failed("doc-1", owner_id="worker-1", error_message="첫 실패")
    repository.mark_indexing("doc-1", owner_id="worker-1", lease_seconds=300)

    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    count = raw_connection.execute(
        "SELECT attempt_count FROM documents WHERE document_id = 'doc-1'"
    ).fetchone()[0]
    raw_connection.close()

    assert count == 2


def test_READY_문서는_제자리_재색인이_거부된다(
    repository: SqliteDocumentRepository,
) -> None:
    """DP-50. 제자리 재색인은 is_active가 1인 채로 검색에서 빠지게 만든다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v1", owner_id="worker-1", chunk_count=5)

    with pytest.raises(MetadataStoreError) as error:
        repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)

    assert "새 버전" in str(error.value)
    document = repository.get("v1")
    assert document is not None
    assert document.status is DocumentStatus.READY


def test_실패한_문서는_다시_색인할_수_있다(repository: SqliteDocumentRepository) -> None:
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_index_failed("v1", owner_id="worker-1", error_message="파싱 실패")

    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)

    document = repository.get("v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEXING


def test_소유자가_아니면_heartbeat이_거부된다(repository: SqliteDocumentRepository) -> None:
    """다른 프로세스에 넘어간 문서를 계속 갱신하면 둘이 같은 문서를 색인하게 된다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)

    with pytest.raises(MetadataStoreError):
        repository.heartbeat("v1", owner_id="worker-2", lease_seconds=300)

    repository.heartbeat("v1", owner_id="worker-1", lease_seconds=300)


def test_리스가_살아있으면_복구가_건드리지_않는다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """heartbeat이 오래됐어도 리스가 남아 있으면 살아 있는 색인으로 본다.

    임베딩 배치 하나가 오래 걸릴 때 살아 있는 프로세스를 죽었다고 오판하지 않게 한다.
    """
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=3600)

    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    raw_connection.execute(
        "UPDATE documents SET heartbeat_at = '2020-01-01T00:00:00+00:00'"
        " WHERE document_id = 'v1'"
    )
    raw_connection.commit()
    raw_connection.close()

    assert repository.recover_abandoned_indexing() == []


def test_계열이_달라도_같은_해시를_찾을_수_있다(
    repository: SqliteDocumentRepository,
) -> None:
    """계열 ID 오타로 같은 파일이 두 번 등록되면 같은 근거가 두 번 검색된다.

    막지는 않는다. 같은 PDF가 두 제품에 공용으로 쓰이는 경우가 있어
    등록을 거부할지는 색인 서비스가 정한다.
    """
    same_hash = "c" * 64
    repository.register(_document(document_id="v1", sha256=same_hash))
    repository.register(
        _document(
            document_id="typo",
            logical_document_id="ls-g100-typo",
            sha256=same_hash,
        )
    )

    found = repository.find_sha256_across_series(same_hash)

    assert [document.document_id for document in found] == ["v1", "typo"]
    assert repository.find_sha256_across_series("d" * 64) == []


def test_파싱_결과가_색인_후에도_남는다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """FR-002. 파싱 직후 메모리에만 있으면 색인이 끝나고 나서 확인할 방법이 없다."""
    repository.register(_document(document_id="v1"))
    repository.record_parse_result("v1", failed_page_count=2, pages_without_text_layer=18)

    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    row = raw_connection.execute(
        "SELECT failed_page_count, pages_without_text_layer FROM documents"
        " WHERE document_id = 'v1'"
    ).fetchone()
    raw_connection.close()

    assert row == (2, 18)


def test_색인_중인_문서는_다시_잡을_수_없다(repository: SqliteDocumentRepository) -> None:
    """두 프로세스가 같은 문서를 동시에 색인하는 것을 막는다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)

    with pytest.raises(MetadataStoreError):
        repository.mark_indexing("v1", owner_id="worker-2", lease_seconds=300)


def test_소유권을_잃으면_결과를_확정할_수_없다(
    repository: SqliteDocumentRepository,
) -> None:
    """복구가 강등한 문서를 원래 프로세스가 READY로 되돌리면 안 된다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.recover_abandoned_indexing(owner_id="worker-1")

    with pytest.raises(MetadataStoreError):
        repository.mark_ready("v1", owner_id="worker-1", chunk_count=5)

    document = repository.get("v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEX_FAILED


def test_같은_소유자로_재기동하면_리스가_남아도_회수한다(
    repository: SqliteDocumentRepository,
) -> None:
    """리스보다 빨리 재기동하면 그 문서가 영구히 갇힌다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=3600)

    assert repository.recover_abandoned_indexing(owner_id="worker-1") == ["v1"]

    document = repository.get("v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEX_FAILED
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)


def test_다른_소유자의_살아있는_색인은_건드리지_않는다(
    repository: SqliteDocumentRepository,
) -> None:
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-2", lease_seconds=3600)

    assert repository.recover_abandoned_indexing(owner_id="worker-1") == []


def test_heartbeat이_리스를_연장한다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """리스 연장이 이 설계의 전제다. 연장이 안 되면 살아 있는 색인이 강등된다."""
    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=10)

    def _lease() -> str:
        raw_connection = sqlite3.connect(tmp_path / "metadata.db")
        value = raw_connection.execute(
            "SELECT lease_until FROM documents WHERE document_id = 'v1'"
        ).fetchone()[0]
        raw_connection.close()
        return value

    before = _lease()
    repository.heartbeat("v1", owner_id="worker-1", lease_seconds=3600)

    assert _lease() > before


def test_색인_중이_아니면_heartbeat이_거부된다(
    repository: SqliteDocumentRepository,
) -> None:
    repository.register(_document(document_id="v1"))

    with pytest.raises(MetadataStoreError):
        repository.heartbeat("v1", owner_id="worker-1", lease_seconds=300)


def test_계열_ID_형식이_틀리면_등록이_거부된다(
    repository: SqliteDocumentRepository,
) -> None:
    with pytest.raises(ValueError, match="문서 계열 ID"):
        repository.register(_document(logical_document_id="manual-G100"))


def test_파싱_결과를_문서로_읽을_수_있다(repository: SqliteDocumentRepository) -> None:
    """FR-002. raw SQL로 DB를 열어야 보인다면 식별 가능하다고 할 수 없다."""
    repository.register(_document(document_id="v1"))
    repository.record_parse_result("v1", failed_page_count=2, pages_without_text_layer=18)

    document = repository.get("v1")
    assert document is not None
    assert document.failed_page_count == 2
    assert document.pages_without_text_layer == 18


def test_상태_갱신_실패_메시지가_원인을_구분한다(
    repository: SqliteDocumentRepository,
) -> None:
    """rowcount 0의 원인은 여럿이다. 고정 문구면 로그만 보고 원인을 반대로 짚는다."""
    with pytest.raises(MetadataStoreError, match="문서 없음"):
        repository.mark_indexing("ghost", owner_id="worker-1", lease_seconds=300)

    repository.register(_document(document_id="v1"))
    repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)
    repository.mark_ready("v1", owner_id="worker-1", chunk_count=1)

    with pytest.raises(MetadataStoreError, match="현재 상태 READY"):
        repository.mark_indexing("v1", owner_id="worker-1", lease_seconds=300)


def test_저장된_모든_시각이_같은_형식이다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """마이크로초 고정 UTC ISO. 형식이 섞이면 lease_until 문자열 비교가 뒤집힌다."""
    import re

    repository.register(_document(document_id="doc-1"))
    repository.mark_indexing("doc-1", owner_id="host-1", lease_seconds=300)
    repository.heartbeat("doc-1", owner_id="host-1", lease_seconds=300)
    repository.mark_ready("doc-1", owner_id="host-1", chunk_count=1)

    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")
    raw_connection = sqlite3.connect(tmp_path / "metadata.db")
    row = raw_connection.execute(
        "SELECT created_at, indexing_started_at, heartbeat_at, indexed_at"
        " FROM documents WHERE document_id = 'doc-1'"
    ).fetchone()
    raw_connection.close()

    for value in row:
        assert value is not None
        assert pattern.match(value), f"형식 불일치: {value}"
