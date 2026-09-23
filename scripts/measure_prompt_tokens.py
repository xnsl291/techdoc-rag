"""근거 예산을 올릴 때 프롬프트 토큰이 num_ctx를 넘는지 잰다.

추정하지 않는다. Ollama `/api/generate`의 `prompt_eval_count`를 읽는다.
"""

import csv
import json
import statistics
import sys
import urllib.request
from pathlib import Path

# 저장소를 어디에 두든 돌아가야 한다. 다른 컴퓨터에서 이어받을 때
# 경로를 고치게 만들지 않는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.query.chat_service import _PROMPT_TEMPLATE  # noqa: E402
from techdoc_rag.query.context_builder import ContextBuilder  # noqa: E402
from techdoc_rag.query.document_scope import (  # noqa: E402
    aliases_by_document,
    scope_from_question,
)

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import build_services  # noqa: E402

BUDGETS = [11500, 14000, 16000, 20000, 26000]


def prompt_tokens(model: str, prompt: str, num_ctx: int) -> int:
    # num_ctx를 안 보내면 Ollama가 기본값 2048로 잘라 버린다. 오류도 안 낸다.
    # 그러면 예산을 아무리 올려도 prompt_eval_count가 2050으로 고정돼 보인다.
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"num_predict": 1, "num_ctx": num_ctx}}
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["prompt_eval_count"]


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    chat, _agent, retriever, _cond = build_services()
    repository = chat._repository
    model = settings.llm.model_name
    num_ctx = settings.llm.runtime_context_tokens
    reserve = settings.llm.generation_max_tokens
    usable = num_ctx - reserve
    print(f"모델 {model} / num_ctx {num_ctx} / 생성 예약 {reserve} / 쓸 수 있는 입력 {usable}")

    with (ROOT / "eval" / "questions.csv").open(encoding="utf-8", newline="") as file:
        questions = [
            row["question"]
            for row in csv.DictReader(file)
            if row["answerable"].strip().upper() == "Y"
        ]
    print(f"답변 가능 {len(questions)}문항으로 잼\n")

    aliases = aliases_by_document(repository)
    # 검색은 한 번만 한다. 예산은 조립 단계 값이라 검색 결과를 재사용할 수 있다.
    retrieved = []
    for q in questions:
        result = retriever.retrieve(q, document_ids=scope_from_question(q, aliases))
        names = {}
        for item in result.chunks:
            did = item.chunk.document_id
            if did not in names:
                doc = repository.get(did)
                if doc is not None:
                    names[did] = doc.original_filename
        retrieved.append((q, result.chunks, names))

    print(f"{'예산':>7} {'근거수 중앙':>10} {'토큰 중앙':>10} {'토큰 최대':>10} {'한도초과':>8}")
    for budget in BUDGETS:
        builder = ContextBuilder(budget_chars=budget)
        counts, tokens = [], []
        for q, chunks, names in retrieved:
            built = builder.build(chunks, names)
            counts.append(len(built.sources))
            tokens.append(
                prompt_tokens(
                    model, _PROMPT_TEMPLATE.format(context=built.text, question=q), num_ctx
                )
            )
        over = sum(t > usable for t in tokens)
        print(
            f"{budget:>7} {statistics.median(counts):>10.1f} "
            f"{statistics.median(tokens):>10.0f} {max(tokens):>10} {over:>8}"
        )


if __name__ == "__main__":
    main()
