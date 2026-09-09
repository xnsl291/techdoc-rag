"""실제 모델과 색인된 문서로 에이전트를 돌려본다 (#42의 실물 항목).

단위 테스트가 못 보는 것을 본다.

1. 모델이 도구를 실제로 고르는가. 어떤 순서로 부르는가
2. 비교 질문에서 문서 두 곳을 다 보는가
3. 없는 제품을 물었을 때 지어내지 않고 확인 불가로 끝나는가
4. 상한에 걸리는 질문이 있는가

사용:
    python scripts/verify_agent_real.py
"""

from __future__ import annotations

import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.ollama_llm_client import OllamaLlmClient  # noqa: E402
from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from techdoc_rag.adapters.sqlite_document_repository import (  # noqa: E402
    SqliteDocumentRepository,
)
from techdoc_rag.agent.agent_service import AgentService  # noqa: E402
from techdoc_rag.agent.tools import ToolBox  # noqa: E402
from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.query.retriever import Retriever  # noqa: E402

QUESTIONS = [
    "어떤 매뉴얼을 갖고 있어?",
    "G100의 정격 전류는?",
    "G100과 M100의 주위 온도 조건을 비교해줘",
    "S9999 제품의 정격 출력은?",  # 없는 제품 — 지어내면 안 됨
]


def main() -> int:
    settings = load_settings()
    repository = SqliteDocumentRepository(settings.storage.metadata_database_path)
    repository.initialize()
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=str(settings.storage.metadata_database_path.parent / "qdrant")),
        collection_name=settings.storage.qdrant_collection_name,
        vector_size=1024,
    )
    vector_store.initialize()
    embedding = OllamaEmbeddingModel(
        model_name=settings.embedding.model_name,
        endpoint=settings.llm.endpoint,
        batch_size=settings.embedding.batch_size,
        num_batch=settings.embedding.max_input_tokens,
        embedding_version="v1",
    )
    llm = OllamaLlmClient(
        model_name=settings.llm.model_name,
        endpoint=settings.llm.endpoint,
        temperature=settings.llm.temperature,
        runtime_context_tokens=settings.llm.runtime_context_tokens,
        thinking_enabled=settings.llm.thinking_enabled,
        max_concurrent_generations=settings.llm.max_concurrent_generations,
        queue_timeout_seconds=settings.llm.queue_timeout_seconds,
        generation_timeout_seconds=settings.llm.generation_timeout_seconds,
    )
    retriever = Retriever(
        embedding_model=embedding,
        vector_store=vector_store,
        repository=repository,
        top_k=settings.retrieval.top_k,
        similarity_threshold=settings.retrieval.similarity_threshold,
        expand_to_neighbors=settings.retrieval.expand_evidence_to_neighbors,
    )
    agent = AgentService(
        llm_client=llm,
        toolbox=ToolBox(retriever=retriever, repository=repository),
        max_steps=settings.agent.max_steps,
        max_answer_tokens=settings.llm.generation_max_tokens,
    )

    print(f"기기: {platform.node()} / 모델: {settings.llm.model_name}")
    print(f"활성 문서: {repository.active_document_ids()}")
    print(f"도구 호출 상한: {settings.agent.max_steps}\n")

    for question in QUESTIONS:
        started = time.perf_counter()
        answer = agent.ask(question)
        elapsed = time.perf_counter() - started
        print(f"Q: {question}")
        limit_note = " / 상한에 걸림" if answer.stopped_at_limit else ""
        print(f"   {elapsed:.1f}s / 도구 {len(answer.steps)}회{limit_note}")
        for index, step in enumerate(answer.steps, start=1):
            print(f"   {index}. {step.tool}({step.arguments})")
        pages = AgentService.evidence_pages(answer.steps)
        if pages:
            print(f"   근거: {', '.join(pages[:6])}")
        print(f"   A: {' '.join(answer.text.split())[:260]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
