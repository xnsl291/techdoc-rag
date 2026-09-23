"""커밋 전 변형 확인 (#50 거절 표현이 섞인 답 표시)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
EV = ROOT / "scripts" / "evaluate.py"

HEDGED = 'return record["answered"] and any(mark in record["text"] for mark in _REFUSAL_MARKS)'

CASES = [
    ("섞인 것을 아예 안 셈", HEDGED, "return False"),
    (
        "이미 거절로 잡힌 것까지 세어 두 번 셈",
        HEDGED,
        'return any(mark in record["text"] for mark in _REFUSAL_MARKS)',
    ),
    (
        "거절 표현을 보면 그냥 거절로 뒤집음",
        HEDGED,
        'record["answered"] = False\n    return True',
    ),
    ("표현 목록에서 '확인할 수 없'을 뺌", '    "확인할 수 없",\n', ""),
    ("표현 목록에서 '포함되어 있지 않'을 뺌", '    "포함되어 있지 않",\n', ""),
    ("표현 목록에서 '명시되어 있지 않'을 뺌", '    "명시되어 있지 않",\n', ""),
    ("표현 목록에서 '나와 있지 않'을 뺌", '    "나와 있지 않",\n', ""),
    (
        "집계에서 거절문구섞임 항목을 뺌",
        '"거절문구섞임": sum(hedged(r) for r in records),',
        '"거절문구섞임": 0,',
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
            EV.write_text(
                original.replace(old, new, 1), encoding="utf-8", newline="\n"
            )
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
