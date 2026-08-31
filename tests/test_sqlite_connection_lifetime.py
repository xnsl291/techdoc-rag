"""커넥션 수명 규칙(DP-54) 검증.

핵심은 두 가지다. FastAPI 워커 스레드 어디서 불러도 ProgrammingError가
나지 않는 것, 그리고 다른 스레드의 쓰기가 끼어들어도 활성 전환 트랜잭션이
온전한 것. 후자는 이슈 #20이 지적한 "check_same_thread=False로 덮으면
조용히 깨지는" 바로 그 지점이다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from techdoc_rag.adapters.sqlite_document_repository import SqliteDocumentRepository
from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.domain.errors import MetadataStoreError


def _document(document_id: str, logical_document_id: str, version: int = 1) -> Document:
    return Document(
        document_id=document_id,
        logical_document_id=logical_document_id,
        document_version=version,
        original_filename="manual.pdf",
        sha256=f"{hash((logical_document_id, version)) & 0xFFFFFFFF:064x}"[:64],
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
    return repo


def test_생성한_스레드가_아닌_곳에서_써도_동작한다(
    repository: SqliteDocumentRepository,
) -> None:
    """저장소는 메인 스레드에서 만들고 모든 작업은 워커 스레드에서 한다.

    커넥션을 인스턴스에 붙드는 구조였다면 첫 호출부터
    ProgrammingError: SQLite objects created in a thread ... 가 난다.
    """
    repository.register(_document("doc-1", "manual-g100"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        loaded = executor.submit(repository.get, "doc-1").result()
        active = executor.submit(repository.active_document_ids).result()

    assert loaded is not None
    assert active == []


def test_여러_스레드가_동시에_읽고_써도_오류가_없다(
    repository: SqliteDocumentRepository,
) -> None:
    """FastAPI 스레드풀(기본 40)을 흉내 낸 혼합 부하."""

    def register_family(index: int) -> None:
        repository.register(_document(f"doc-{index}", f"manual-{index}"))
        repository.mark_indexing(f"doc-{index}", owner_id=f"host-{index}", lease_seconds=300)
        repository.mark_ready(f"doc-{index}", owner_id=f"host-{index}", chunk_count=1)
        repository.activate(f"doc-{index}")

    def read_repeatedly(_: int) -> None:
        for _ in range(20):
            repository.active_document_ids()
            repository.get("doc-0")

    with ThreadPoolExecutor(max_workers=16) as executor:
        writers = [executor.submit(register_family, index) for index in range(8)]
        readers = [executor.submit(read_repeatedly, index) for index in range(8)]
        for future in writers + readers:
            future.result()  # 예외가 있었다면 여기서 다시 던져진다

    assert len(repository.active_document_ids()) == 8


def test_활성_전환_중_다른_스레드의_쓰기가_끼어들어도_원자적이다(
    repository: SqliteDocumentRepository, tmp_path: Path
) -> None:
    """v1 활성 상태에서 v2로 전환하는 동안 다른 스레드가 쓰기를 계속한다.

    전환의 두 UPDATE(구버전 비활성 + 신버전 활성) 사이에 다른 커밋이 끼면
    active가 0개이거나 2개인 순간이 관측된다. 커넥션이 작업 단위로 분리되어
    있으면 각 전환은 자기 커넥션의 트랜잭션 안에서만 끝난다.
    """
    repository.register(_document("v1", "manual-g100", version=1))
    repository.mark_indexing("v1", owner_id="host", lease_seconds=300)
    repository.mark_ready("v1", owner_id="host", chunk_count=1)
    repository.activate("v1")

    repository.register(_document("v2", "manual-g100", version=2))
    repository.mark_indexing("v2", owner_id="host", lease_seconds=300)
    repository.mark_ready("v2", owner_id="host", chunk_count=1)

    def noisy_writer(index: int) -> None:
        # 전환과 무관한 계열에 쓰기를 계속 일으켜 커밋이 겹치게 한다.
        repository.register(_document(f"noise-{index}", f"noise-{index}"))
        repository.mark_indexing(f"noise-{index}", owner_id="noise", lease_seconds=300)
        repository.heartbeat(f"noise-{index}", owner_id="noise", lease_seconds=300)

    with ThreadPoolExecutor(max_workers=8) as executor:
        noise = [executor.submit(noisy_writer, index) for index in range(12)]
        switch = executor.submit(repository.activate, "v2")
        for future in [switch, *noise]:
            future.result()

    family = [repository.get("v1"), repository.get("v2")]
    active_in_family = [
        document for document in family if document is not None and document.is_active
    ]
    assert len(active_in_family) == 1
    assert active_in_family[0].document_id == "v2"
    v1 = repository.get("v1")
    assert v1 is not None
    assert v1.status is DocumentStatus.INACTIVE


def test_없는_경로의_부모는_만들고_잠긴_DB는_MetadataStoreError(
    tmp_path: Path,
) -> None:
    """연결 실패가 sqlite3.Error 그대로 새지 않고 저장소 예외로 감싸진다."""
    nested = SqliteDocumentRepository(tmp_path / "a" / "b" / "metadata.db")
    nested.initialize()  # 부모 디렉터리 생성 확인

    # 디렉터리를 DB 경로로 주면 열기 자체가 실패한다.
    broken = SqliteDocumentRepository(tmp_path / "dir_as_db")
    (tmp_path / "dir_as_db").mkdir(exist_ok=True)
    with pytest.raises(MetadataStoreError):
        broken.initialize()
