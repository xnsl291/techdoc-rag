"""커밋 전 변형 확인 (#53 근거 선정 드러내기)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CB = ROOT / "src" / "techdoc_rag" / "query" / "context_builder.py"
CS = ROOT / "src" / "techdoc_rag" / "query" / "chat_service.py"
SCH = ROOT / "src" / "techdoc_rag" / "api" / "schemas.py"

CASES = [
    (
        "밀려난 것을 기록하지 않음",
        CB,
        "                dropped.append(result)\n                continue",
        "                continue",
    ),
    (
        "중복도 밀려난 것으로 셈",
        CB,
        "            if result.chunk.chunk_id in seen:\n"
        "                continue  # 중복은 밀려난 것이 아니라 같은 것이므로 dropped에 넣지 않는다",
        "            if result.chunk.chunk_id in seen:\n"
        "                dropped.append(result)\n"
        "                continue",
    ),
    (
        "밀려난 것을 인용에 안 담음",
        CS,
        "        citations += [\n"
        "            to_citation(result.chunk, False, False) for result in context.dropped\n"
        "        ]",
        "        citations += []",
    ),
    (
        "밀려난 것을 프롬프트에 간 것으로 표시",
        CS,
        "            to_citation(result.chunk, False, False) for result in context.dropped",
        "            to_citation(result.chunk, False, True) for result in context.dropped",
    ),
    (
        "밀려난 것을 답변이 쓴 것으로 표시",
        CS,
        "            to_citation(result.chunk, False, False) for result in context.dropped",
        "            to_citation(result.chunk, True, False) for result in context.dropped",
    ),
    (
        "API가 reached_prompt를 항상 True로 보냄",
        SCH,
        "                    reached_prompt=citation.reached_prompt,",
        "                    reached_prompt=True,",
    ),
]


def run_tests() -> bool:
    result = subprocess.run(
        [str(PY), "-m", "pytest", "tests/", "-q", "--no-header", "-x"],
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
    for label, path, old, new in CASES:
        original = path.read_text(encoding="utf-8")
        try:
            if old not in original:
                print(f"[패턴없음] {label}")
                failures.append(label)
                continue
            path.write_text(
                original.replace(old, new, 1), encoding="utf-8", newline="\n"
            )
            passed = run_tests()
        finally:
            path.write_text(original, encoding="utf-8", newline="\n")
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
