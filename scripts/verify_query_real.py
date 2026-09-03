"""실물 매뉴얼을 색인하고 실제 질문을 넣어 본다 (#19의 실물 항목).

Step 6의 완료 기준이 여기서 확인된다 — **질문 하나에 답과 Citation이 나온다.**

단위 테스트가 못 보는 것을 본다.

1. 실물 파이프라인 전체(색인→활성화→검색→조립→생성)에서 답과 출처 페이지가 나오는가
2. 4B 모델이 [번호] 인용 표기를 실제로 따르는가 — grounding v1(인용 번호 방식)의 전제
3. 매뉴얼과 무관한 질문이 근거 있는 답으로 둔갑하지 않는가
4. 질문 하나의 체감 시간(이 기기 값. 기기 종속)

사용:
    python scripts/verify_query_real.py --pdf <M100 사용설명서 PDF>
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.ollama_llm_client import OllamaLlmClient  # noqa: E402
from techdoc_rag.adapters.pypdfium_parser import PypdfiumParser  # noqa: E402
from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from techdoc_rag.adapters.sqlite_document_repository import (  # noqa: E402
    SqliteDocumentRepository,
)
from techdoc_rag.ingestion.chunker import RecursiveChunker  # noqa: E402
from techdoc_rag.ingestion.ingestion_service import IngestionService  # noqa: E402
from techdoc_rag.query.chat_service import ChatService  # noqa: E402
from techdoc_rag.query.context_builder import ContextBuilder  # noqa: E402
from techdoc_rag.query.retriever import Retriever  # noqa: E402

# 매뉴얼에 답이 있는 질문(공개 Q&A 유형을 흉내 낸 것. 평가셋 문항 아님)과
# 매뉴얼로 답할 수 없는 질문 하나.
QUESTIONS = [
    "인버터 설치 시 주위 온도 조건은?",
    "과전류 트립이 발생하는 원인은 무엇인가?",
    "김치찌개를 맛있게 끓이는 방법은?",  # 무관 질문 — 근거 있는 답이 나오면 안 됨
]


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--pdf", type=Path, required=True)
    argument_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    arguments = argument_parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="verify_query_"))
    print(f"기기: {platform.node()} / 작업 공간: {workspace}")

    repository = SqliteDocumentRepository(workspace / "metadata.db")
    repository.initialize()
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=str(workspace / "qdrant")),
        collection_name="techdoc-bge-m3-v1",
        vector_size=1024,
    )
    vector_store.initialize()
    embedding = OllamaEmbeddingModel(
        model_name="bge-m3",
        endpoint=arguments.endpoint,
        batch_size=128,
        num_batch=2048,
        embedding_version="v1",
    )

    # 색인 + 활성화 (질의 서비스는 활성 문서만 본다)
    ingestion = IngestionService(
        parser=PypdfiumParser(),
        chunker=RecursiveChunker(size_chars=1200, overlap_chars=150, config_version="v1"),
        repository=repository,
        embedding_model=embedding,
        vector_store=vector_store,
        owner_id=platform.node(),
        lease_seconds=300,
    )
    indexed = ingestion.ingest(arguments.pdf, "ls-m100", document_version=1, document_type="manual")
    repository.activate(indexed.document_id)
    print(f"색인·활성화: {indexed.document_id} (청크 {indexed.chunk_count}개)\n")

    llm = OllamaLlmClient(
        model_name="qwen3.5:4b-q4_K_M",
        endpoint=arguments.endpoint,
        temperature=0.0,
        runtime_context_tokens=8192,
        thinking_enabled=False,
        max_concurrent_generations=1,
        queue_timeout_seconds=120,
        generation_timeout_seconds=300,
    )
    service = ChatService(
        retriever=Retriever(
            embedding_model=embedding,
            vector_store=vector_store,
            repository=repository,
            top_k=5,
            similarity_threshold=0.0,  # [미확정, 시작점] settings.yaml과 동일
        ),
        context_builder=ContextBuilder(budget_chars=6000),  # [미확정, 시작점] top_k×최대청크+여유
        llm_client=llm,
        repository=repository,
        max_answer_tokens=512,
    )

    ungrounded_ok = True
    for index, question in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        answer = service.ask(question)
        elapsed = time.perf_counter() - started
        used = [c for c in answer.citations if c.is_used_in_answer]
        print(f"[{index}] Q: {question}")
        print(f"    {elapsed:.1f}s / 판정: {answer.no_answer_reason or '답변됨'}")
        print(f"    A: {answer.text.strip()[:300]}")
        if used:
            pages = ", ".join(
                f"{c.display_name} p.{c.page_start}"
                + (f"~{c.page_end}" if c.page_end != c.page_start else "")
                for c in used
            )
            print(f"    근거: {pages}")
        print()
        if "김치찌개" in question:
            if answer.is_answered:
                ungrounded_ok = False  # 무관 질문이 근거 있는 답으로 둔갑함
        elif not answer.is_answered:
            ungrounded_ok = False  # 답이 있어야 하는 질문이 No-answer로 빠짐 (회귀)

    if not ungrounded_ok:
        print("실패: 무관 질문이 답변됨 또는 답변 가능 질문이 No-answer로 판정됨")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
