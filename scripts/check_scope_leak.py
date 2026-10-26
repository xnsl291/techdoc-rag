"""문서 범위를 지정했을 때 범위 밖 문서의 근거가 섞이지 않는지 실물로 잰다 (#43).

단위 테스트는 가짜 저장소로 "지정이 자동을 이긴다"까지만 본다. 실제 Qdrant와
이웃 청크 확장을 거쳐도 범위 밖 문서가 안 들어오는지는 여기서 본다.

1. 검색 계층: 실제 질문 전부 x (문서 단독 범위 전부 + 두 권 묶음 하나)
2. /chat 전체 경로: 문서마다 질문 2건, 응답 citations까지

사용:
    python scripts/check_scope_leak.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from evaluate import build_services  # noqa: E402


def main() -> int:
    chat, _agent, retriever, _conditions = build_services()
    docs = list(chat._repository.active_document_ids())
    with (ROOT / "eval" / "questions_real.csv").open(encoding="utf-8", newline="") as file:
        questions = [(row["id"], row["question"]) for row in csv.DictReader(file)]
    print(f"활성 문서 {len(docs)}권, 질문 {len(questions)}건")

    leaks = []
    total = 0
    scopes = [[doc] for doc in docs] + [docs[:2]]
    for question_id, question in questions:
        for scope in scopes:
            for item in retriever.retrieve(question, document_ids=scope).chunks:
                total += 1
                if item.chunk.document_id not in scope:
                    leaks.append((question_id, scope, item.chunk.document_id))
    print(
        f"[검색] 범위 {len(scopes)}종 x 질문 {len(questions)}건, "
        f"청크 {total}개, 범위 밖 {len(leaks)}개"
    )

    asked = 0
    cited = 0
    empty = 0
    for index, doc in enumerate(docs):
        for question_id, question in questions[index * 2 : index * 2 + 2]:
            answer = chat.ask(question, document_ids=[doc])
            asked += 1
            if not answer.citations:
                empty += 1
            for citation in answer.citations:
                cited += 1
                if citation.document_id != doc:
                    leaks.append((question_id, [doc], citation.document_id))
    print(f"[chat] 질문 {asked}건, 근거 {cited}개(근거 0건 응답 {empty})")

    for leak in leaks[:10]:
        print("  범위 밖:", leak)
    print(f"범위 밖 합계 {len(leaks)}개")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
