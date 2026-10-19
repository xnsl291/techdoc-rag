"""운영 설정에서 프롬프트가 num_ctx를 넘는 문항이 몇 건이고, 넘으면 답이 바뀌는지 잰다 (#57).

2026-09-22 측정은 만든 질문 중 답변 가능 24건만 봐서 최대 7,830 토큰이었다.
실제 질문과 답변 불가 문항은 재지 않았다. 여기서는 두 평가셋 전 문항을 본다.

1. 운영과 같은 순서로 프롬프트를 조립하고(자동 범위, 검색, 근거 예산) 넉넉한
   num_ctx로 보내 실제 토큰 수를 읽는다
2. 운영 num_ctx를 넘는 문항만 골라 운영 값과 16384로 각각 답을 받아 비교한다

Ollama는 넘치면 맨 앞 4토큰만 남기고 뒤쪽 절반을 남긴다(probe_truncation.py).
프롬프트 맨 앞의 "근거에 없는 내용은 답하지 마라"가 그때 사라진다.

사용:
    python scripts/probe_context_overflow.py
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from probe_answers import build_services  # noqa: E402

from techdoc_rag.config import load_settings  # noqa: E402
from techdoc_rag.query.chat_service import _PROMPT_TEMPLATE  # noqa: E402
from techdoc_rag.query.document_scope import (  # noqa: E402
    aliases_by_document,
    scope_from_question,
)

WIDE_CTX = 16384


def prompt_tokens(endpoint: str, model: str, prompt: str) -> int:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": 1, "num_ctx": 32768},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["prompt_eval_count"]


def load_questions() -> list[tuple[str, str, str]]:
    rows = []
    for name in ("questions.csv", "questions_real.csv"):
        with (ROOT / "eval" / name).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                rows.append((row["id"], row["answerable"].strip().upper(), row["question"]))
    return rows


def main() -> int:
    settings = load_settings()
    chat, _agent, _retriever, _conditions = build_services()
    num_ctx = settings.llm.runtime_context_tokens
    usable = num_ctx - settings.llm.generation_max_tokens
    aliases = aliases_by_document(chat._repository)

    measured = []
    for question_id, answerable, question in load_questions():
        scope = scope_from_question(question, aliases)
        retrieval = chat._retriever.retrieve(question, document_ids=scope)
        if not retrieval.chunks:
            continue
        names = chat._display_names(retrieval.chunks)
        context = chat._context_builder.build(retrieval.chunks, names)
        prompt = _PROMPT_TEMPLATE.format(context=context.text, question=question)
        tokens = prompt_tokens(settings.llm.endpoint, settings.llm.model_name, prompt)
        measured.append((question_id, answerable, question, tokens, len(prompt)))

    over_ctx = [m for m in measured if m[3] > num_ctx]
    over_usable = [m for m in measured if m[3] > usable]
    print(f"num_ctx {num_ctx}, 생성 예약 뺀 입력 한도 {usable}")
    print(f"문항 {len(measured)}건, 최대 {max(m[3] for m in measured)} 토큰")
    print(f"  num_ctx 초과(앞이 잘림) {len(over_ctx)}건")
    print(f"  입력 한도 초과(생성 자리 부족) {len(over_usable)}건")
    for question_id, answerable, _q, tokens, chars in sorted(over_usable, key=lambda m: -m[3]):
        print(f"    {question_id} 답변가능={answerable} {tokens} 토큰 {chars}자")

    print(f"\nnum_ctx를 넘은 문항을 {num_ctx}와 {WIDE_CTX}로 다시 답하게 한다")
    llm = chat._llm_client
    for question_id, answerable, question, tokens, _chars in over_ctx:
        results = []
        for ctx in (num_ctx, WIDE_CTX):
            llm._runtime_context_tokens = ctx
            answer = chat.ask(question)
            results.append(answer)
        llm._runtime_context_tokens = num_ctx
        print(f"== {question_id} 답변가능={answerable} {tokens} 토큰  {question[:40]}")
        for ctx, answer in zip((num_ctx, WIDE_CTX), results, strict=True):
            used = sum(c.is_used_in_answer for c in answer.citations)
            head = answer.text[:110].replace("\n", " ")
            print(f"   ctx {ctx:>5}: answered={answer.is_answered} 인용={used}  {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
