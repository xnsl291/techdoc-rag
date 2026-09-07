"""고정 질문 묶음으로 답변을 뽑아 파일로 남긴다 — 변경 전후 비교용.

**평가셋이 아니다.** 정답 라벨이 없으므로 점수를 매기지 않는다. 하는 일은
"같은 질문에 어떤 근거를 골라 뭐라고 답했는가"를 같은 조건에서 두 번 재서
**바뀐 것을 눈에 보이게** 하는 것뿐이다(규칙 L: 고치기 전에 기준선을 잰다).

질문 묶음은 실물에서 관측된 문제를 담는다.
- 같은 뜻을 다르게 물었을 때 답이 갈리는가 (2026-09-07에 갈리는 것을 확인)
- 답이 잘 나오던 질문이 변경 후에도 그대로인가 (회귀 확인)
- 무관한 질문을 여전히 거절하는가

사용:
    python scripts/probe_answers.py --label baseline
    (설정·프롬프트를 바꾼 뒤)
    python scripts/probe_answers.py --label prompt-v2
    python scripts/probe_answers.py --compare baseline prompt-v2
"""

from __future__ import annotations

import argparse
import json
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
from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.query.chat_service import PROMPT_VERSION, ChatService  # noqa: E402
from techdoc_rag.query.context_builder import ContextBuilder  # noqa: E402
from techdoc_rag.query.query_expander import QueryExpander  # noqa: E402
from techdoc_rag.query.retriever import Retriever  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parents[1] / "data" / "probes"

# 같은 뜻을 다르게 물은 묶음(group)과, 회귀를 보려는 단독 질문.
QUESTIONS = [
    ("정격출력", "정격출력이 얼마에요?"),
    ("정격출력", "정격 출력 알려줘"),
    ("정격출력", "이 인버터 출력 용량이 어떻게 되나요?"),
    ("설치온도", "인버터 설치 시 주위 온도 조건은?"),
    ("설치온도", "설치할 때 온도 몇 도까지 괜찮아요?"),
    ("과전류", "과전류 트립이 발생하는 원인은 무엇인가?"),
    ("무관", "김치찌개를 맛있게 끓이는 방법은?"),
]


def _build_service() -> tuple[ChatService, dict]:
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
    service = ChatService(
        retriever=Retriever(
            embedding_model=embedding,
            vector_store=vector_store,
            repository=repository,
            top_k=settings.retrieval.top_k,
            similarity_threshold=settings.retrieval.similarity_threshold,
            query_expander=(
                QueryExpander(
                    llm_client=llm,
                    short_query_chars=settings.retrieval.short_query_chars,
                    max_variants=settings.retrieval.max_query_variants,
                )
                if settings.retrieval.expand_short_queries
                else None
            ),
        ),
        context_builder=ContextBuilder(budget_chars=settings.retrieval.context_budget_chars),
        llm_client=llm,
        repository=repository,
        max_answer_tokens=settings.llm.generation_max_tokens,
    )
    # 무엇으로 낸 결과인지 함께 남긴다. 이게 없으면 두 결과의 차이가
    # 코드 때문인지 설정 때문인지 나중에 구분할 수 없다(NFR-004).
    conditions = {
        "machine": platform.node(),
        "llm_model": settings.llm.model_name,
        "embedding_model": settings.embedding.model_name,
        "top_k": settings.retrieval.top_k,
        "similarity_threshold": settings.retrieval.similarity_threshold,
        "context_budget_chars": settings.retrieval.context_budget_chars,
        "generation_max_tokens": settings.llm.generation_max_tokens,
        "prompt_version": PROMPT_VERSION,
    }
    return service, conditions


def run(label: str) -> int:
    service, conditions = _build_service()
    print(f"조건: {json.dumps(conditions, ensure_ascii=False)}")
    records = []
    for group, question in QUESTIONS:
        started = time.perf_counter()
        answer = service.ask(question)
        elapsed = time.perf_counter() - started
        used = [
            f"p.{c.page_start}~{c.page_end}" for c in answer.citations if c.is_used_in_answer
        ]
        records.append(
            {
                "group": group,
                "question": question,
                "verdict": answer.no_answer_reason or "ANSWERED",
                "text": answer.text.strip(),
                "used_pages": used,
                "all_pages": [f"p.{c.page_start}~{c.page_end}" for c in answer.citations],
                "seconds": round(elapsed, 2),
            }
        )
        print(f"  [{group}] {question}")
        print(f"      {records[-1]['verdict']} / {elapsed:.1f}s / 근거 {used}")
        print(f"      {answer.text.strip()[:110]}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"{label}.json"
    with out.open("w", encoding="utf-8") as file:
        json.dump(
            {"conditions": conditions, "records": records}, file, ensure_ascii=False, indent=2
        )
    print(f"\n저장: {out}")
    _summarize(records)
    return 0


def _summarize(records: list[dict]) -> None:
    """같은 뜻 묶음에서 근거가 얼마나 일치하는지. 낮으면 표현에 흔들린다는 뜻."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    print("\n묶음별 근거 일치")
    for group, items in groups.items():
        if len(items) < 2:
            continue
        page_sets = [set(item["used_pages"]) for item in items]
        shared = set.intersection(*page_sets) if all(page_sets) else set()
        print(f"  {group}: {len(items)}개 질문, 공통 근거 {sorted(shared) or '없음'}")


def compare(left: str, right: str) -> int:
    def load(label: str) -> dict:
        with (RESULT_DIR / f"{label}.json").open(encoding="utf-8") as file:
            return json.load(file)

    a, b = load(left), load(right)
    changed = {
        key: (a["conditions"][key], b["conditions"][key])
        for key in a["conditions"]
        if a["conditions"][key] != b["conditions"].get(key)
    }
    print(f"바뀐 조건: {json.dumps(changed, ensure_ascii=False) if changed else '없음'}\n")
    for old, new in zip(a["records"], b["records"], strict=True):
        marks = []
        if old["verdict"] != new["verdict"]:
            marks.append(f"판정 {old['verdict']}→{new['verdict']}")
        if old["used_pages"] != new["used_pages"]:
            marks.append(f"근거 {old['used_pages']}→{new['used_pages']}")
        if old["text"] != new["text"]:
            marks.append("답변 문구 바뀜")
        print(f"[{new['group']}] {new['question']}")
        print(f"   {' / '.join(marks) if marks else '변화 없음'}")
        if old["text"] != new["text"]:
            print(f"   전: {old['text'][:90]}")
            print(f"   후: {new['text'][:90]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="이번 실행 결과를 저장할 이름")
    parser.add_argument("--compare", nargs=2, metavar=("이전", "이후"))
    arguments = parser.parse_args()
    if arguments.compare:
        return compare(*arguments.compare)
    if not arguments.label:
        parser.error("--label 또는 --compare 중 하나가 필요함")
    return run(arguments.label)


if __name__ == "__main__":
    sys.exit(main())
