"""실제 매뉴얼 PDF 하나를 처음부터 끝까지 색인해 본다 (#17의 실물 항목).

단위 테스트가 못 보는 것을 본다.

1. 실물 PDF가 등록→파싱→청킹→임베딩→저장→READY까지 통과하는가, 몇 분 걸리는가
2. Qdrant 벡터 수가 chunk_count와 같은가
3. 같은 chunk_id 재저장이 개수를 늘리지 않는가 (chunk_id→point_id 결정론)
4. 이전 실행의 잔여 벡터를 주입했을 때 delete_stale_runs가 그것만 지우는가
5. 같은 파일을 다시 넣으면 색인 없이 중복으로 판정되는가

Qdrant는 로컬 파일 모드(DP-52), SQLite는 임시 경로를 쓴다 — 실행할 때마다
빈 상태에서 시작해 결과가 이전 실행에 오염되지 않는다.

사용:
    python scripts/verify_ingestion_real.py --pdf <매뉴얼 PDF> [--logical-id ls-m100]
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402
from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from techdoc_rag.adapters.sqlite_document_repository import (  # noqa: E402
    SqliteDocumentRepository,
)
from techdoc_rag.domain.document import DocumentStatus  # noqa: E402
from techdoc_rag.domain.indexing import IndexRun  # noqa: E402
from techdoc_rag.ingestion.chunker import RecursiveChunker  # noqa: E402
from techdoc_rag.ingestion.ingestion_service import IngestionService  # noqa: E402

VECTOR_SIZE = 1024  # bge-m3


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--pdf", type=Path, required=True)
    argument_parser.add_argument("--logical-id", default="ls-m100")
    argument_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    arguments = argument_parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="verify_ingestion_"))
    print(f"기기: {platform.node()} / 작업 공간: {workspace}")

    repository = SqliteDocumentRepository(workspace / "metadata.db")
    repository.initialize()
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=str(workspace / "qdrant")),
        collection_name="techdoc-bge-m3-v1",
        vector_size=VECTOR_SIZE,
    )
    vector_store.initialize()
    embedding = OllamaEmbeddingModel(
        model_name="bge-m3",
        endpoint=arguments.endpoint,
        batch_size=128,
        num_batch=2048,
        embedding_version="v1",
    )
    parser = PypdfiumParser()
    chunker = RecursiveChunker(size_chars=1200, overlap_chars=150, config_version="v1")
    service = IngestionService(
        parser=parser,
        chunker=chunker,
        repository=repository,
        embedding_model=embedding,
        vector_store=vector_store,
        owner_id=platform.node(),
        lease_seconds=300,
    )

    # 1) 실물 색인 한 바퀴
    started = time.perf_counter()
    result = service.ingest(
        arguments.pdf, arguments.logical_id, document_version=1, document_type="manual"
    )
    elapsed = time.perf_counter() - started
    document = repository.get(result.document_id)
    assert document is not None
    print(
        f"\n[1] 색인: {result.document_id} / 청크 {result.chunk_count}개"
        f" / {elapsed:.0f}s ({result.chunk_count / elapsed:.1f} chunk/s, 이 기기 값)"
        f" / 상태 {document.status.value} / 페이지 {document.page_count}"
        f" (실패 {document.failed_page_count}, 텍스트 없음 {document.pages_without_text_layer})"
    )
    assert document.status is DocumentStatus.READY

    # 2) 벡터 수 = 청크 수
    count = vector_store.count()
    verdict = "통과" if count == result.chunk_count else "실패"
    print(f"\n[2] 벡터 수: {count} (기대 {result.chunk_count}) → {verdict}")
    if count != result.chunk_count:
        return 1  # 이후 단계가 전부 이 개수를 기준으로 하므로 즉시 끝낸다

    failures: list[str] = []

    # 이후 단계에서 쓸 현재 run id — 저장된 payload에서 꺼낸다(정본은 저장소).
    points, _ = vector_store._client.scroll(  # noqa: SLF001 (검증 스크립트 한정)
        collection_name="techdoc-bge-m3-v1", limit=1, with_payload=True
    )
    current_run_id = points[0].payload["index_run_id"]

    # 3) 같은 chunk_id 재저장 → 개수 불변 (덮어쓰기 확인)
    parsed = parser.parse(arguments.pdf)
    chunks = chunker.chunk(parsed, result.document_id, 1)[:8]
    vectors = embedding.embed_documents([chunk.text for chunk in chunks])
    same_run = IndexRun(
        document_id=result.document_id,
        logical_document_id=arguments.logical_id,
        document_type="manual",
        index_run_id=current_run_id,
    )
    vector_store.upsert(chunks, vectors, same_run)
    after_reupsert = vector_store.count()
    verdict = "통과" if after_reupsert == count else "실패 (중복 적재)"
    if after_reupsert != count:
        failures.append("[3] 재저장 멱등")
    print(f"\n[3] 같은 chunk_id 재저장 후: {after_reupsert} (기대 {count}) → {verdict}")

    # 4) 잔여 벡터 주입 → 정리 → 원래 개수 복귀
    #    이전 실행에서 청크가 더 많았던 상황을 흉내 낸다: 지금은 없는 chunk_id들.
    stale_chunks = [
        replace(chunk, chunk_id=f"{result.document_id}:{9000 + index:04d}")
        for index, chunk in enumerate(chunks)
    ]
    stale_run = IndexRun(
        document_id=result.document_id,
        logical_document_id=arguments.logical_id,
        document_type="manual",
        index_run_id="00000000000000000000000000000000",
    )
    vector_store.upsert(stale_chunks, vectors, stale_run)
    inflated = vector_store.count()
    vector_store.delete_stale_runs(result.document_id, current_run_id)
    restored = vector_store.count()
    verdict = "통과" if (inflated == count + len(stale_chunks) and restored == count) else "실패"
    if verdict != "통과":
        failures.append("[4] 잔여 정리")
    print(
        f"\n[4] 잔여 주입 {inflated} → delete_stale_runs → {restored}"
        f" (기대 {count}) → {verdict}"
    )

    # 5) 같은 파일 재투입 → 중복 판정, 색인 안 함
    started = time.perf_counter()
    again = service.ingest(
        arguments.pdf, arguments.logical_id, document_version=2, document_type="manual"
    )
    verdict = "통과" if again.duplicate_of == result.document_id else "실패"
    if verdict != "통과":
        failures.append("[5] 중복 판정")
    print(
        f"\n[5] 같은 파일 재투입: duplicate_of={again.duplicate_of}"
        f" / {time.perf_counter() - started:.2f}s → {verdict}"
    )

    if failures:
        # print만 하고 exit 0이면 자동 게이트에서 실패가 통과로 보인다(리뷰 N-2).
        print(f"\n실패한 단계: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
