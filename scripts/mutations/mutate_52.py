"""커밋 전 변형 확인 (#52 제품명으로 검색 범위 좁히기)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DS = ROOT / "src" / "techdoc_rag" / "query" / "document_scope.py"
CS = ROOT / "src" / "techdoc_rag" / "query" / "chat_service.py"

CASES = [
    (
        "경계 검사를 빼고 그냥 부분 문자열로 봄",
        DS,
        'pattern = rf"(?<![0-9A-Za-z]){re.escape(alias)}(?![0-9A-Za-z])"',
        'pattern = rf"{re.escape(alias)}"',
    ),
    (
        "경계를 ASCII가 아니라 \\w로 봄 (한글이 낱말 문자로 잡힘)",
        DS,
        'pattern = rf"(?<![0-9A-Za-z]){re.escape(alias)}(?![0-9A-Za-z])"',
        'pattern = rf"(?<!\\w){re.escape(alias)}(?!\\w)"',
    ),
    (
        "대소문자 무시를 뺌",
        DS,
        "return re.search(pattern, question, re.IGNORECASE) is not None",
        "return re.search(pattern, question) is not None",
    ),
    (
        "못 찾았을 때 None 대신 빈 목록을 돌려줌",
        DS,
        "return matched or None",
        "return matched",
    ),
    (
        "이름을 끝 조각이 아니라 논리 id 통째로 씀",
        DS,
        'alias = document.logical_document_id.rsplit("-", 1)[-1]',
        "alias = document.logical_document_id",
    ),
    (
        "구한 범위를 검색기에 안 넘김",
        CS,
        "retrieval = self._retriever.retrieve(question, document_ids=scope)",
        "retrieval = self._retriever.retrieve(question)",
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
