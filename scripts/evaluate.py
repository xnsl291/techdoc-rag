"""평가셋으로 검색·근거·거절 정확도를 잰다.

`probe_answers.py`와 하는 일이 다르다. 프로브는 고치기 전후를 **비교**하고
여기는 **정답과 대조**한다. 정답이 있으니 "몇 퍼센트 맞았다"를 말할 수 있다.

재는 것 넷이다. 앞의 셋은 자동으로 세고, 넷째는 사람이 판정한다.

1. **검색 적중** 정답 페이지가 검색 결과에 있었나
2. **근거 적중** 그 페이지를 답변에 실제로 썼나(인용 번호로 확인, DP-56)
3. **거절 정확** 답할 수 없는 질문을 거절했나, 그리고 답할 수 있는 질문을
   괜히 거절하지는 않았나. 뒤쪽을 같이 보지 않으면 "다 거절하면 만점"이 된다
4. **답변-근거 일치** 답이 근거와 맞나. 이건 매뉴얼을 아는 사람이 읽어야 한다.
   같은 모델에게 채점시키면 틀린 답을 낸 쪽이 자기 답을 채점하는 셈이라 안 한다

**장애를 낮은 점수로 감추지 않는다.** 검색·생성 실패는 예외로 그대로 올린다
(D-005). 평가 도중 Ollama가 죽었는데 "정확도 0%"로 기록되면 원인을 잘못 짚는다.

사용:
    python scripts/evaluate.py --check
    python scripts/evaluate.py --label baseline
    python scripts/evaluate.py --label baseline --apply-judge
    python scripts/evaluate.py --compare baseline topk-8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_answers import build_services  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.csv"
RESULT_DIR = ROOT / "data" / "evals"

# "ls-m100-v1 p.248-250" 또는 "ls-m100-v1 p.42"
_EVIDENCE = re.compile(r"^(?P<document>[\w\-.]+)\s+p\.(?P<start>\d+)(?:-(?P<end>\d+))?$")


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    answerable: bool
    # (문서 id, 시작 쪽, 끝 쪽). answerable이 False면 비어 있다.
    evidence: tuple[tuple[str, int, int], ...]
    note: str


def _shown(path: Path) -> str:
    """저장소 안이면 짧게, 밖이면 그대로. relative_to는 밖이면 ValueError를 낸다."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_evidence(text: str) -> list[tuple[str, int, int]]:
    """`ls-m100-v1 p.248-250; ls-g100-v1 p.369-370`을 나눈다."""
    spans = []
    for piece in text.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        matched = _EVIDENCE.match(piece)
        if matched is None:
            raise ValueError(f"evidence 형식이 아님: {piece!r} (예: ls-m100-v1 p.248-250)")
        start = int(matched["start"])
        end = int(matched["end"] or start)
        if end < start:
            raise ValueError(f"끝 쪽이 시작보다 작음: {piece!r}")
        spans.append((matched["document"], start, end))
    return spans


def load_questions(path: Path, page_counts: dict[str, int]) -> list[Question]:
    """평가셋을 읽고 그 자리에서 검사한다.

    읽기와 검사를 나누지 않는 이유는 오타가 실행 전에 걸려야 하기 때문이다.
    40문항을 다 돌린 뒤에 "문서 이름이 틀렸다"를 알면 20분을 버린다.

    문제를 모아서 한 번에 알린다. 하나 고치고 다시 돌리기를 반복하지 않도록.
    """
    if not path.exists():
        raise FileNotFoundError(f"평가셋이 없음: {path}")

    questions: list[Question] = []
    problems: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            identifier = (row.get("id") or "").strip()
            text = (row.get("question") or "").strip()
            if not identifier or not text:
                continue  # 채우다 만 줄은 건너뛴다

            flag = (row.get("answerable") or "").strip().upper()
            if flag not in {"Y", "N"}:
                problems.append(f"{identifier}: answerable이 Y도 N도 아님 ({flag!r})")
                continue
            answerable = flag == "Y"

            raw = (row.get("evidence") or "").strip()
            try:
                spans = parse_evidence(raw)
            except ValueError as error:
                problems.append(f"{identifier}: {error}")
                continue

            if answerable and not spans:
                problems.append(f"{identifier}: answerable=Y인데 evidence가 비었음")
                continue
            if not answerable and spans:
                problems.append(f"{identifier}: answerable=N인데 evidence가 있음")
                continue

            for document, _start, end in spans:
                if document not in page_counts:
                    problems.append(
                        f"{identifier}: 색인되지 않은 문서 {document!r}"
                        f" (있는 것: {', '.join(sorted(page_counts))})"
                    )
                elif end > page_counts[document]:
                    problems.append(
                        f"{identifier}: {document}는 {page_counts[document]}쪽까지인데"
                        f" {end}쪽을 가리킴"
                    )

            questions.append(
                Question(
                    id=identifier,
                    question=text,
                    answerable=answerable,
                    evidence=tuple(spans),
                    note=(row.get("note") or "").strip(),
                )
            )

    if problems:
        raise ValueError("평가셋에 문제가 있음\n  " + "\n  ".join(problems))
    if not questions:
        raise ValueError(f"채워진 질문이 없음: {path}")
    return questions


def overlaps(span: tuple[str, int, int], document: str, start: int, end: int) -> bool:
    """페이지 범위가 겹치는지. 여기가 틀리면 모든 지표가 틀린다.

    정답이 248~250이고 검색 결과가 245~248이면 248이 겹치므로 적중으로 본다.
    청크가 페이지 경계를 걸쳐 있어 정확히 일치하는 일이 드물다.
    """
    return span[0] == document and span[1] <= end and start <= span[2]


def score(records: list[dict]) -> dict:
    answerable = [r for r in records if r["answerable"]]
    refusable = [r for r in records if not r["answerable"]]
    seconds = [r["seconds"] for r in records]

    return {
        "문항": len(records),
        "답변가능": len(answerable),
        "답변불가": len(refusable),
        "검색적중": sum(r["retrieval_hit"] for r in answerable),
        "근거적중": sum(r["citation_hit"] for r in answerable),
        "거절정확": sum(not r["answered"] for r in refusable),
        "오거절": sum(not r["answered"] for r in answerable),
        "응답시간_중앙값": round(statistics.median(seconds), 2) if seconds else 0.0,
        "응답시간_최대": round(max(seconds), 2) if seconds else 0.0,
    }


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total}  ({hit / total * 100:.1f}%)" if total else f"{hit}/0  (해당 없음)"


def print_summary(summary: dict, judged: dict | None = None) -> None:
    print()
    print(
        f"문항 {summary['문항']}개 "
        f"(답변가능 {summary['답변가능']} / 답변불가 {summary['답변불가']})"
    )
    print(f"  검색 적중   {_ratio(summary['검색적중'], summary['답변가능'])}")
    print(f"  근거 적중   {_ratio(summary['근거적중'], summary['답변가능'])}")
    print(f"  거절 정확   {_ratio(summary['거절정확'], summary['답변불가'])}")
    print(f"  오거절      {_ratio(summary['오거절'], summary['답변가능'])}")
    if judged:
        print(f"  답변 일치   {_ratio(judged['맞음'], judged['판정됨'])}  (사람 판정)")
    else:
        print("  답변 일치   판정 전")
    print(
        f"  응답시간    중앙값 {summary['응답시간_중앙값']}초"
        f" / 최대 {summary['응답시간_최대']}초"
    )


def write_judge_sheet(path: Path, records: list[dict]) -> None:
    """답변이 근거와 맞는지 사람이 적을 표.

    judge 칸에 O 또는 X만 적으면 --apply-judge가 집계한다. 답변가능 문항만 넣는다.
    거절해야 할 문항은 위 지표에서 이미 자동으로 센다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["id", "question", "정답위치", "시스템이_쓴_근거", "답변", "judge", "메모"]
        )
        for record in records:
            if not record["answerable"]:
                continue
            writer.writerow(
                [
                    record["id"],
                    record["question"],
                    "; ".join(record["expected"]),
                    "; ".join(record["used_pages"]) or "(없음)",
                    " ".join(record["text"].split()),
                    "",
                    "",
                ]
            )


def read_judge(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"판정 표가 없음: {path}")
    judged = correct = 0
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            mark = (row.get("judge") or "").strip().upper()
            if mark in {"O", "X"}:
                judged += 1
                correct += mark == "O"
    return {"판정됨": judged, "맞음": correct}


def run(label: str, questions_path: Path) -> int:
    # 평가셋을 먼저 읽고 검사한다. 40문항을 다 돌린 뒤에 오타를 알면 20분을 버린다.
    questions = load_questions(questions_path, _page_counts())
    service, _agent, conditions = build_services()
    conditions["questions"] = questions_path.name
    print(f"평가셋: {_shown(questions_path)} ({len(questions)}문항)")
    print(f"조건: {json.dumps(conditions, ensure_ascii=False)}")
    print()

    records = []
    for question in questions:
        started = time.perf_counter()
        answer = service.ask(question.question)
        elapsed = time.perf_counter() - started

        searched = [(c.document_id, c.page_start, c.page_end) for c in answer.citations]
        used = [
            (c.document_id, c.page_start, c.page_end)
            for c in answer.citations
            if c.is_used_in_answer
        ]
        retrieval_hit = any(
            overlaps(span, *found) for span in question.evidence for found in searched
        )
        citation_hit = any(
            overlaps(span, *found) for span in question.evidence for found in used
        )

        records.append(
            {
                "id": question.id,
                "question": question.question,
                "answerable": question.answerable,
                "expected": [f"{d} p.{s}-{e}" for d, s, e in question.evidence],
                "answered": answer.is_answered,
                "no_answer_reason": (
                    answer.no_answer_reason.value if answer.no_answer_reason else None
                ),
                "retrieval_hit": retrieval_hit,
                "citation_hit": citation_hit,
                "searched_pages": [f"{d} p.{s}-{e}" for d, s, e in searched],
                "used_pages": [f"{d} p.{s}-{e}" for d, s, e in used],
                "text": answer.text.strip(),
                "seconds": round(elapsed, 2),
            }
        )

        if question.answerable:
            mark = "적중" if retrieval_hit else "놓침"
            mark += "/근거O" if citation_hit else "/근거X"
            if not answer.is_answered:
                mark = f"오거절({answer.no_answer_reason.value})"
        else:
            mark = "거절함" if not answer.is_answered else "답해버림"
        print(f"  {question.id} [{mark}] {elapsed:.1f}s  {question.question[:44]}")

    summary = score(records)
    print_summary(summary)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / f"{label}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"conditions": conditions, "summary": summary, "records": records},
            file,
            ensure_ascii=False,
            indent=2,
        )
    judge_path = RESULT_DIR / f"{label}_judge.csv"
    write_judge_sheet(judge_path, records)

    print()
    print(f"저장: {_shown(result_path)}")
    print(f"사람 판정용: {_shown(judge_path)}")
    print("  judge 칸에 O 또는 X를 적은 뒤 --apply-judge로 다시 집계함")
    return 0


def apply_judge(label: str) -> int:
    result_path = RESULT_DIR / f"{label}.json"
    with result_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    judged = read_judge(RESULT_DIR / f"{label}_judge.csv")
    payload["judged"] = judged
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print_summary(payload["summary"], judged)
    print()
    print(f"갱신: {_shown(result_path)}")
    return 0


def compare(left: str, right: str) -> int:
    def load(label: str) -> dict:
        with (RESULT_DIR / f"{label}.json").open(encoding="utf-8") as file:
            return json.load(file)

    before, after = load(left), load(right)
    changed = {
        key: (before["conditions"][key], after["conditions"][key])
        for key in before["conditions"]
        if before["conditions"][key] != after["conditions"].get(key)
    }
    print(f"바뀐 조건: {json.dumps(changed, ensure_ascii=False) if changed else '없음'}")
    print()
    print(f"{'지표':16} {left:>14} {right:>14}")
    for key in ("검색적중", "근거적중", "거절정확", "오거절", "응답시간_중앙값"):
        print(f"{key:16} {before['summary'][key]:>14} {after['summary'][key]:>14}")

    # 문항별로 뒤집힌 것을 보인다. 총계만 같아도 안에서 바뀌었을 수 있다.
    old = {r["id"]: r for r in before["records"]}
    print()
    print("문항별 변화")
    flipped = 0
    for record in after["records"]:
        previous = old.get(record["id"])
        if previous is None:
            print(f"  {record['id']} [새 문항] {record['question'][:40]}")
            continue
        marks = []
        if previous["retrieval_hit"] != record["retrieval_hit"]:
            direction = "놓침에서 적중" if record["retrieval_hit"] else "적중에서 놓침"
            marks.append(f"검색 적중 {direction}")
        if previous["citation_hit"] != record["citation_hit"]:
            marks.append("근거 적중 " + ("X에서 O" if record["citation_hit"] else "O에서 X"))
        if previous["answered"] != record["answered"]:
            marks.append("거절 여부 바뀜")
        if marks:
            flipped += 1
            print(f"  {record['id']} {' / '.join(marks)}  {record['question'][:36]}")
    if not flipped:
        print("  뒤집힌 문항 없음")
    return 0


def _page_counts() -> dict[str, int]:
    from techdoc_rag.adapters.sqlite_document_repository import SqliteDocumentRepository
    from techdoc_rag.config import load_settings

    settings = load_settings()
    repository = SqliteDocumentRepository(settings.storage.metadata_database_path)
    repository.initialize()
    counts = {}
    for document_id in repository.active_document_ids():
        document = repository.get(document_id)
        if document is not None:
            counts[document_id] = document.page_count
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="평가셋만 검사하고 끝냄")
    parser.add_argument("--label", help="이번 실행 결과를 저장할 이름")
    parser.add_argument("--apply-judge", action="store_true", help="판정 표를 읽어 다시 집계")
    parser.add_argument("--compare", nargs=2, metavar=("이전", "이후"))
    # DP-32가 Dev와 Holdout을 나누라고 정해 둠. 같은 셋으로 튜닝하고 최종 보고까지
    # 하면 그 셋에 과적합된다.
    parser.add_argument(
        "--questions", type=Path, default=QUESTIONS_PATH, help="평가셋 파일 경로"
    )
    arguments = parser.parse_args()

    if arguments.compare:
        return compare(*arguments.compare)
    if not arguments.check and not arguments.label:
        parser.error("--check, --label, --compare 중 하나가 필요함")

    # 평가셋 문제는 사람이 고칠 것이라 스택트레이스가 아니라 문구로 보인다.
    # 장애(Ollama·저장소)는 여기서 잡지 않고 그대로 올린다(D-005).
    try:
        if arguments.check:
            page_counts = _page_counts()
            questions = load_questions(arguments.questions, page_counts)
            answerable = sum(q.answerable for q in questions)
            print(
                f"이상 없음. {len(questions)}문항 "
                f"(답변가능 {answerable} / 답변불가 {len(questions) - answerable})"
            )
            documents = ", ".join(f"{k}({v}쪽)" for k, v in sorted(page_counts.items()))
            print(f"색인된 문서: {documents}")
            return 0
        if arguments.apply_judge:
            return apply_judge(arguments.label)
        return run(arguments.label, arguments.questions)
    except (ValueError, FileNotFoundError) as error:
        print(f"\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
