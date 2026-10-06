"""커밋 전 변형 확인 (#65 거절 판정이 양방향으로 틀리는 것)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
EV = ROOT / "scripts" / "evaluate.py"

SILENT = (
    'return not record["answered"] and not any(\n'
    '        mark in record["text"] for mark in _REFUSAL_MARKS\n'
    "    )"
)

CASES = [
    ("거울 방향을 아예 안 셈", SILENT, "return False"),
    (
        "이미 답함으로 잡힌 것까지 세어 두 번 셈",
        SILENT,
        'return not any(mark in record["text"] for mark in _REFUSAL_MARKS)',
    ),
    (
        "거절 표현이 없으면 그냥 답한 것으로 뒤집음",
        SILENT,
        'record["answered"] = True\n    return True',
    ),
    (
        "집계에서 인용없이답함 항목을 뺌",
        '"인용없이답함": sum(answered_without_citation(r) for r in records),',
        '"인용없이답함": 0,',
    ),
    (
        "판정표에서 답변 불가 문항을 건너뜀",
        "for record in sorted(records, key=lambda r: not suspect(r)):",
        "for record in sorted(\n"
        "            [r for r in records if r['answerable']], key=lambda r: not suspect(r)\n"
        "        ):",
    ),
    (
        "판정표의 라벨 열을 한 값으로 고정",
        '"답변가능" if answerable else "답변불가",',
        '"답변가능",',
    ),
    (
        "판정 집계를 라벨 구분 없이 한 숫자로 합침",
        'if (row.get("라벨") or "답변가능").strip() == "답변불가":',
        "if False:",
    ),
    (
        "의심 표시를 한 종류만 함",
        "            if answered_without_citation(record):\n"
        '                return "인용없이답함"\n',
        "",
    ),
]


def run_tests() -> bool:
    result = subprocess.run(
        [str(PY), "-m", "pytest", "tests/test_evaluate_hedged.py", "-q", "--no-header"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def main() -> int:
    if not run_tests():
        print("기준 상태부터 실패함. 중단")
        return 1

    failures = []
    for label, old, new in CASES:
        original = EV.read_text(encoding="utf-8")
        try:
            if old not in original:
                print(f"[패턴없음] {label}")
                failures.append(label)
                continue
            EV.write_text(original.replace(old, new, 1), encoding="utf-8", newline="\n")
            passed = run_tests()
        finally:
            EV.write_text(original, encoding="utf-8", newline="\n")
        if passed:
            print(f"[공회전] {label} -> 깨뜨렸는데 테스트가 통과함")
            failures.append(label)
        else:
            print(f"[검출] {label}")

    if not run_tests():
        print("복원 후 실패함. 파일을 확인할 것")
        return 1
    print()
    print(f"검출 {len(CASES) - len(failures)}/{len(CASES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
