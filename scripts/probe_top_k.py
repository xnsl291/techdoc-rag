"""top_k를 키우면 정답 쪽이 검색 후보에 들어오는지만 본다.

답변은 만들지 않는다. LLM을 부르지 않으므로 빠르다.

이 실험이 가르는 것
- 들어온다  -> 검색은 찾고 있는데 상위에서 밀린 것. Reranker 도입 근거가 생김
- 안 들어온다 -> 검색 자체(임베딩·청킹)의 문제. Reranker를 넣어도 소용없음
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# 저장소를 어디에 두든 돌아가야 한다. 다른 컴퓨터에서 이어받을 때
# 경로를 고치게 만들지 않는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_questions, overlaps  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel  # noqa: E402
from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from techdoc_rag.adapters.sqlite_document_repository import (  # noqa: E402
    SqliteDocumentRepository,
)
from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.query.retriever import Retriever  # noqa: E402

K_VALUES = [8, 16, 30, 50]


def main() -> int:
    settings = load_settings()
    repository = SqliteDocumentRepository(settings.storage.metadata_database_path)
    repository.initialize()
    page_counts = {}
    for document_id in repository.active_document_ids():
        document = repository.get(document_id)
        if document is not None:
            page_counts[document_id] = document.page_count

    questions = [q for q in load_questions(ROOT / "eval" / "questions.csv", page_counts)
                 if q.answerable]

    store = QdrantVectorStore(
        client=QdrantClient(path=str(settings.storage.metadata_database_path.parent / "qdrant")),
        collection_name=settings.storage.qdrant_collection_name,
        vector_size=1024,
    )
    store.initialize()
    embedding = OllamaEmbeddingModel(
        model_name=settings.embedding.model_name,
        endpoint=settings.llm.endpoint,
        batch_size=settings.embedding.batch_size,
        num_batch=settings.embedding.max_input_tokens,
        embedding_version="v1",
    )

    print(f"답변가능 {len(questions)}문항 / 문서 {len(page_counts)}권")
    print(f"현재 설정: top_k={settings.retrieval.top_k}, "
          f"이웃 확장={settings.retrieval.expand_evidence_to_neighbors}")
    print()

    # 문항별로 정답이 몇 번째에 처음 나오는지. k=50 기준.
    big = Retriever(
        embedding_model=embedding, vector_store=store, repository=repository,
        top_k=50, similarity_threshold=settings.retrieval.similarity_threshold,
        expand_to_neighbors=settings.retrieval.expand_evidence_to_neighbors,
    )
    first_rank: dict[str, int | None] = {}
    print(f"{'문항':6} {'정답 첫 등장 순위':>16}  질문")
    for question in questions:
        result = big.retrieve(question.question)
        rank = None
        for index, item in enumerate(result.chunks, start=1):
            chunk = item.chunk
            if any(overlaps(span, chunk.document_id, chunk.page_start, chunk.page_end)
                   for span in question.evidence):
                rank = index
                break
        first_rank[question.id] = rank
        shown = f"{rank}위" if rank else "50개 안에 없음"
        print(f"{question.id:6} {shown:>16}  {question.question[:40]}")

    print()
    print("top_k별 검색 적중")
    for k in K_VALUES:
        hit = sum(1 for r in first_rank.values() if r is not None and r <= k)
        print(f"  top_k={k:>3}   {hit}/{len(questions)}  ({hit / len(questions) * 100:.1f}%)")

    missing = [qid for qid, r in first_rank.items() if r is None]
    if missing:
        print()
        print(f"50개 안에도 없는 문항 {len(missing)}건: {', '.join(missing)}")
        print("  이건 top_k나 Reranker로 못 고침. 검색 자체의 문제임")
    return 0


if __name__ == "__main__":
    sys.exit(main())
