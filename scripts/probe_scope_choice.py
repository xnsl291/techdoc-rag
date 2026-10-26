"""되묻기(화면에서 "어느 제품인가요?")가 실제로 얼마나 도움이 되는지 잰다.

제품을 특정하지 않은 실제 질문은 검색 적중이 27.3%다(2026-10-01). 같은 질문에
사용자가 자기 제품을 골랐다고 가정하고 그 문서로 범위를 좁혀 다시 돌린다.

## 무엇을 가정하나

이 질문들은 정답 문서가 2권 이상이다. 제품을 특정하지 않았으니 어느 제품의
매뉴얼에도 답이 있다는 뜻이다. **사용자는 자기 제품을 알고 있으므로 그중 하나를
고른다.** 그래서 정답 문서 각각으로 범위를 좁혀 한 번씩 돌리고, 결과를 평균한다.

평균을 쓰는 이유: 사용자가 어느 제품을 가졌는지는 질문마다 다르다. "가장 잘
나온 문서"만 고르면 실제보다 좋게 나온다.

## 무엇을 가정하지 않나

사용자가 틀린 제품을 고르는 경우는 재지 않는다. 그건 화면이 막을 일이다.
되묻기 화면이 정답 후보 제품만 보여 주기 때문이다(질문에 걸린 문서들).

사용:
    python scripts/probe_scope_choice.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from evaluate import build_services, overlaps, parse_evidence  # noqa: E402


def main() -> int:
    chat, _agent, _retriever, _conditions = build_services()

    with (ROOT / "eval" / "questions_real.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    targets = []
    for row in rows:
        if row["answerable"].strip().upper() != "Y":
            continue
        spans = parse_evidence(row["evidence"])
        documents = sorted({span[0] for span in spans})
        if len(documents) > 1:  # 제품을 특정하지 않은 질문
            targets.append((row["id"], row["question"], spans, documents))

    print(f"제품 미특정 질문 {len(targets)}건. 정답 문서 각각으로 좁혀 다시 돌린다\n")

    per_question = []
    for question_id, question, spans, documents in targets:
        hits = []
        for document in documents:
            answer = chat.ask(question, document_ids=[document])
            found = [
                (c.document_id, c.page_start, c.page_end) for c in answer.citations
            ]
            reached = [
                (c.document_id, c.page_start, c.page_end)
                for c in answer.citations
                if c.reached_prompt
            ]
            used = [
                (c.document_id, c.page_start, c.page_end)
                for c in answer.citations
                if c.is_used_in_answer
            ]
            # 고른 문서의 정답 쪽만 본다. 다른 문서의 정답은 범위 밖이라 나올 수 없다.
            mine = [span for span in spans if span[0] == document]
            hits.append(
                (
                    any(overlaps(s, *f) for s in mine for f in found),
                    any(overlaps(s, *f) for s in mine for f in reached),
                    any(overlaps(s, *f) for s in mine for f in used),
                    answer.is_answered,
                )
            )
        retrieval = statistics.mean(h[0] for h in hits)
        prompt = statistics.mean(h[1] for h in hits)
        citation = statistics.mean(h[2] for h in hits)
        per_question.append((retrieval, prompt, citation))
        print(
            f"  {question_id}  문서 {len(documents)}권  "
            f"검색 {retrieval:.2f}  프롬프트 {prompt:.2f}  근거 {citation:.2f}  "
            f"{question[:34]}"
        )

    n = len(per_question)
    print()
    print(f"제품을 골랐을 때 ({n}문항, 정답 문서별 평균)")
    print(f"  검색 적중     {statistics.mean(q[0] for q in per_question) * 100:.1f}%")
    print(f"  프롬프트 도달 {statistics.mean(q[1] for q in per_question) * 100:.1f}%")
    print(f"  근거 적중     {statistics.mean(q[2] for q in per_question) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
