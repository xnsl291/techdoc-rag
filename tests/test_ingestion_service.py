"""색인 서비스 테스트 (#17).

저장소는 실물 SQLite를 쓴다 — 상태 전이 규칙(NEW→INDEXING→READY, 소유권,
리스)이 저장소에 박혀 있어 가짜로 바꾸면 그 규칙을 안 지나가는 테스트가 된다.
파서·임베딩·벡터 저장소는 가짜로 바꿔 호출 순서와 실패 경로를 관측한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from techdoc_rag.adapters.sqlite_document_repository import SqliteDocumentRepository
from techdoc_rag.domain.chunk import Chunk
from techdoc_rag.domain.document import DocumentStatus
from techdoc_rag.domain.errors import IndexingError, ParsingError
from techdoc_rag.domain.indexing import IndexRun
from techdoc_rag.domain.parsing import PageText, ParsedDocument
from techdoc_rag.ingestion.chunker import RecursiveChunker
from techdoc_rag.ingestion.ingestion_service import IngestionService


class FakeParser:
    """페이지 내용을 주입받는 파서. parse 호출 대상 경로를 기록한다."""

    parser_version = "fake-parser-1"

    def __init__(self, pages: Sequence[str], failed_pages: tuple[int, ...] = ()) -> None:
        self._pages = pages
        self._failed_pages = failed_pages
        self.parsed_paths: list[Path] = []

    def parse(self, pdf_path: Path) -> ParsedDocument:
        self.parsed_paths.append(pdf_path)
        return ParsedDocument(
            pages=tuple(
                PageText(page_no=index + 1, text=text, has_text_layer=bool(text.strip()))
                for index, text in enumerate(self._pages)
            ),
            failed_pages=self._failed_pages,
        )


class FailingParser:
    parser_version = "fake-parser-1"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        raise ParsingError("문서를 열 수 없음")


class FakeEmbeddingModel:
    """batch_size=2로 작게 잡아 배치 반복이 실제로 일어나게 한다."""

    model_name = "fake-embed"
    dimension = 4
    embedding_version = "v1"
    max_input_tokens = 2048
    batch_size = 2

    def __init__(self, events: list[str], fail_on_batch: int | None = None) -> None:
        self._events = events
        self._fail_on_batch = fail_on_batch
        self._batch_index = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._batch_index += 1
        if self._fail_on_batch == self._batch_index:
            raise IndexingError("임베딩 서버 응답 없음")
        self._events.append(f"embed:{len(texts)}")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.upserted: list[tuple[list[str], IndexRun]] = []
        self.stale_cleanups: list[tuple[str, str]] = []

    def initialize(self) -> None:
        pass

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], index_run: IndexRun
    ) -> None:
        self._events.append(f"upsert:{len(chunks)}")
        self.upserted.append(([chunk.chunk_id for chunk in chunks], index_run))

    def search(self, query_vector, top_k, active_document_ids):
        return []

    def delete_document(self, document_id: str) -> None:
        pass

    def delete_stale_runs(self, document_id: str, current_index_run_id: str) -> None:
        self._events.append("cleanup")
        self.stale_cleanups.append((document_id, current_index_run_id))

    def count(self) -> int:
        return sum(len(chunk_ids) for chunk_ids, _ in self.upserted)


class RecordingRepository(SqliteDocumentRepository):
    """호출 순서를 관측하기 위한 얇은 스파이. 동작은 실물 그대로다."""

    def __init__(self, database_path: Path, events: list[str]) -> None:
        super().__init__(database_path)
        self._events = events

    def heartbeat(self, document_id: str, owner_id: str, lease_seconds: int) -> None:
        self._events.append("heartbeat")
        super().heartbeat(document_id, owner_id, lease_seconds)

    def mark_ready(self, document_id: str, owner_id: str, chunk_count: int, **kwargs) -> None:
        self._events.append("ready")
        super().mark_ready(document_id, owner_id, chunk_count, **kwargs)


# 1200자 청크 기준으로 청크 4개 이상이 나오도록 페이지를 채운다(배치 2 → 배치 2회 이상).
_PAGES = ["가나다라마바사 " * 200, "아자차카타파하 " * 200, "인버터 과전류 보호 " * 150]


@pytest.fixture()
def events() -> list[str]:
    return []


@pytest.fixture()
def repository(tmp_path: Path, events: list[str]) -> RecordingRepository:
    repo = RecordingRepository(tmp_path / "metadata.db", events)
    repo.initialize()
    return repo


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "manual.pdf"
    path.write_bytes("%PDF-fake 내용은 파서가 가짜라 안 읽는다".encode())
    return path


def _service(
    repository: RecordingRepository,
    events: list[str],
    parser=None,
    embedding=None,
    vector_store=None,
) -> tuple[IngestionService, FakeVectorStore]:
    store = vector_store or FakeVectorStore(events)
    service = IngestionService(
        parser=parser or FakeParser(_PAGES),
        chunker=RecursiveChunker(size_chars=1200, overlap_chars=150, config_version="v1"),
        repository=repository,
        embedding_model=embedding or FakeEmbeddingModel(events),
        vector_store=store,
        owner_id="test-host",
        lease_seconds=300,
    )
    return service, store


def test_색인이_끝나면_READY이고_재현성_정보가_남는다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    service, _ = _service(repository, events)

    result = service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    assert result.document_id == "ls-g100-v1"
    assert result.chunk_count >= 4  # 배치(2개) 반복이 실제로 일어날 만큼
    document = repository.get("ls-g100-v1")
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert document.is_active is False  # 활성 전환은 색인 서비스의 일이 아니다
    assert document.page_count == 3  # 등록 시 0 → 파싱 실측으로 채워짐
    assert document.parser_version == "fake-parser-1"
    assert document.chunk_config_version == "v1"
    assert document.embedding_model == "fake-embed"
    assert document.embedding_version == "v1"


def test_배치마다_저장_직후_heartbeat가_온다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    """임베딩이 오래 걸려도 리스가 만료되지 않으려면 문서 단위가 아니라
    배치 단위로 갱신되어야 한다. 정리와 READY는 모든 배치 뒤에만 온다."""
    service, _ = _service(repository, events)

    service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    batch_events = [event for event in events if event != "ready"]
    assert batch_events[-1] == "cleanup"
    pattern = batch_events[:-1]
    assert len(pattern) % 3 == 0 and len(pattern) >= 6  # (embed, upsert, heartbeat) 2회 이상
    for index in range(0, len(pattern), 3):
        assert pattern[index].startswith("embed:")
        assert pattern[index + 1].startswith("upsert:")
        assert pattern[index + 2] == "heartbeat"
    assert events[-1] == "ready"


def test_모든_배치가_같은_index_run을_쓰고_그_run으로_정리한다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    service, store = _service(repository, events)

    service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    run_ids = {index_run.index_run_id for _, index_run in store.upserted}
    assert len(run_ids) == 1  # 배치마다 run이 갈리면 정리 단계가 자기 벡터를 지운다
    assert store.stale_cleanups == [("ls-g100-v1", run_ids.pop())]


def test_파싱_실패는_INDEX_FAILED로_남고_예외가_올라온다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    service, store = _service(repository, events, parser=FailingParser())

    with pytest.raises(ParsingError):
        service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    document = repository.get("ls-g100-v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEX_FAILED
    assert store.upserted == []


def test_임베딩_실패는_INDEX_FAILED로_남는다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    embedding = FakeEmbeddingModel(events, fail_on_batch=2)
    service, _ = _service(repository, events, embedding=embedding)

    with pytest.raises(IndexingError, match="응답 없음"):
        service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    document = repository.get("ls-g100-v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEX_FAILED


def test_텍스트가_없는_문서는_INDEX_FAILED다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    service, store = _service(repository, events, parser=FakeParser(["", "  "]))

    with pytest.raises(IndexingError, match="색인할 텍스트 없음"):
        service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    document = repository.get("ls-g100-v1")
    assert document is not None
    assert document.status is DocumentStatus.INDEX_FAILED
    assert document.page_count == 2  # 파싱까지는 됐으므로 결과는 남는다
    assert store.upserted == []


def test_같은_계열_같은_해시는_다시_색인하지_않는다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    service, store = _service(repository, events)
    first = service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")
    upserts_after_first = len(store.upserted)

    second = service.ingest(pdf, "ls-g100", document_version=2, document_type="manual")

    assert second.duplicate_of == first.document_id
    assert len(store.upserted) == upserts_after_first  # 임베딩·저장이 다시 일어나지 않음
    assert repository.get("ls-g100-v2") is None  # 새 버전으로 등록되지도 않음


def test_실패한_문서는_같은_파일로_재시도하면_이어서_색인한다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    """리뷰 B-1. 중복 사전 확인이 INDEX_FAILED까지 잡으면 일과성 실패
    (임베딩 서버 다운 등)가 같은 파일로는 영영 복구되지 않는다."""
    failing_service, _ = _service(repository, events, parser=FailingParser())
    with pytest.raises(ParsingError):
        failing_service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    service, store = _service(repository, events)
    result = service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    assert result.duplicate_of is None  # 실패 문서를 "이미 처리됨"으로 오인하지 않음
    assert result.document_id == "ls-g100-v1"  # 새 버전이 아니라 같은 자리에서 재개
    document = repository.get("ls-g100-v1")
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert len(store.upserted) >= 1


def test_READY_문서는_같은_파일_재시도에서도_중복이다(
    repository: RecordingRepository, events: list[str], pdf: Path
) -> None:
    """재시도 허용은 NEW/INDEX_FAILED뿐이다. 성공한 문서까지 다시 색인하면
    재실행할 때마다 색인 비용을 다시 치른다."""
    service, store = _service(repository, events)
    first = service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")
    upserts = len(store.upserted)

    second = service.ingest(pdf, "ls-g100", document_version=1, document_type="manual")

    assert second.duplicate_of == first.document_id
    assert len(store.upserted) == upserts
