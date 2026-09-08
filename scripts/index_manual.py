"""매뉴얼을 설정된 저장소에 색인하고 활성화한다 (#42).

`verify_ingestion_real.py`와 다른 점은 저장 위치다. 그쪽은 실행할 때마다 임시
폴더를 만들어 검증만 하고 버린다. 이 스크립트는 `config/settings.yaml`이 가리키는
실제 저장소에 넣어 **화면과 API가 바로 쓸 수 있게** 한다.

문서 간 비교를 시연하려면 같은 저장소에 두 권 이상이 있어야 한다.

사용:
    python scripts/index_manual.py --pdf <매뉴얼.pdf> --id ls-g100
    python scripts/index_manual.py --list
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402
from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from techdoc_rag.adapters.sqlite_document_repository import (  # noqa: E402
    SqliteDocumentRepository,
)
from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.ingestion.chunker import RecursiveChunker  # noqa: E402
from techdoc_rag.ingestion.ingestion_service import IngestionService  # noqa: E402

VECTOR_SIZE = 1024  # bge-m3


def _build(settings):
    repository = SqliteDocumentRepository(settings.storage.metadata_database_path)
    repository.initialize()
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=str(settings.storage.metadata_database_path.parent / "qdrant")),
        collection_name=settings.storage.qdrant_collection_name,
        vector_size=VECTOR_SIZE,
    )
    vector_store.initialize()
    return repository, vector_store


def show(settings) -> int:
    repository, vector_store = _build(settings)
    active = repository.active_document_ids()
    print(f"활성 문서 {len(active)}개, 벡터 {vector_store.count():,}개")
    for document_id in active:
        document = repository.get(document_id)
        if document is None:
            continue
        # chunk_count는 DB에 있으나 Document 모델로 읽어오지 않는다. 여기서는
        # 문서를 알아볼 수 있으면 되므로 쪽수와 색인 시각만 보인다.
        indexed = document.indexed_at.strftime("%Y-%m-%d %H:%M") if document.indexed_at else "-"
        print(
            f"  {document_id:20} {document.original_filename[:44]:46}"
            f" {document.page_count:>4}쪽  색인 {indexed}"
        )
    return 0


def index(settings, pdf_path: Path, logical_id: str, version: int) -> int:
    repository, vector_store = _build(settings)
    service = IngestionService(
        parser=PypdfiumParser(),
        chunker=RecursiveChunker(
            size_chars=settings.chunking.size_chars,
            overlap_chars=settings.chunking.overlap_chars,
            config_version=settings.chunking.config_version,
        ),
        repository=repository,
        embedding_model=OllamaEmbeddingModel(
            model_name=settings.embedding.model_name,
            endpoint=settings.llm.endpoint,
            batch_size=settings.embedding.batch_size,
            num_batch=settings.embedding.max_input_tokens,
            embedding_version="v1",
        ),
        vector_store=vector_store,
        owner_id=platform.node(),
        lease_seconds=settings.indexing.lease_seconds,
    )

    started = time.perf_counter()
    result = service.ingest(pdf_path, logical_id, document_version=version, document_type="manual")
    elapsed = time.perf_counter() - started

    if result.duplicate_of is not None:
        print(f"이미 있는 문서임: {result.duplicate_of} (같은 계열에 같은 해시)")
        return 0

    # 색인과 활성화를 나눠 둔 것은 운영자가 READY를 확인한 뒤 노출하기 위함이다
    # (01 §17.6). 여기서는 바로 이어서 한다 — 개인이 쓰는 색인 도구이기 때문이다.
    repository.activate(result.document_id)
    print(
        f"색인·활성화: {result.document_id} / 청크 {result.chunk_count}개 /"
        f" {elapsed:.0f}초 ({result.chunk_count / elapsed:.1f} chunk/s, 이 기기 값)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--id", help="문서 계열 ID. 소문자와 하이픈만 (예: ls-g100)")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="색인된 문서 보기")
    arguments = parser.parse_args()

    settings = load_settings()
    if arguments.list:
        return show(settings)
    if not arguments.pdf or not arguments.id:
        parser.error("--pdf와 --id가 필요함 (또는 --list)")
    if not arguments.pdf.is_file():
        parser.error(f"파일이 없음: {arguments.pdf}")
    return index(settings, arguments.pdf, arguments.id, arguments.version)


if __name__ == "__main__":
    sys.exit(main())
