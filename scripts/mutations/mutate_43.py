"""커밋 전 변형 확인 (#43 /chat에 문서 범위 지정)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CS = ROOT / "src" / "techdoc_rag" / "query" / "chat_service.py"
APP = ROOT / "src" / "techdoc_rag" / "api" / "app.py"
SCH = ROOT / "src" / "techdoc_rag" / "api" / "schemas.py"

CASES = [
    (
        "지정을 무시하고 늘 자동 범위를 씀",
        CS,
        "        if document_ids is not None:\n"
        "            scope = self._checked_scope(document_ids)",
        "        if False:\n"
        "            scope = self._checked_scope(document_ids)",
    ),
    (
        "지정과 자동 범위의 합집합을 씀",
        CS,
        "            scope = self._checked_scope(document_ids)\n",
        "            scope = self._checked_scope(document_ids) + (\n"
        "                scope_from_question(\n"
        "                    question, aliases_by_document(self._repository)\n"
        "                ) or []\n"
        "            )\n",
    ),
    (
        "없는 문서를 조용히 걸러 냄",
        CS,
        "        if unknown:\n            raise UnknownDocumentError(unknown)",
        "        if False:\n            raise UnknownDocumentError(unknown)",
    ),
    (
        "중복을 안 합침",
        CS,
        "        return list(dict.fromkeys(document_ids))",
        "        return document_ids",
    ),
    (
        "API가 범위를 서비스에 안 넘김",
        APP,
        "answer = chat_service.ask(request.question, document_ids=request.document_ids)",
        "answer = chat_service.ask(request.question)",
    ),
    (
        "없는 문서를 503으로 냄",
        APP,
        "            raise HTTPException(status_code=404, detail=str(error)) from error\n"
        "        except (RetrievalError",
        "            raise HTTPException(status_code=503, detail=str(error)) from error\n"
        "        except (RetrievalError",
    ),
    (
        "빈 목록을 받아 줌",
        SCH,
        "document_ids: list[str] | None = Field(default=None, min_length=1)",
        "document_ids: list[str] | None = Field(default=None)",
    ),
]


# 변형마다 그 변형을 지키려고 쓴 테스트 하나만 돌린다. 전체를 돌리면 다른
# 테스트가 우연히 잡아 줘서, 정작 그 자리를 지키는 테스트가 없어도 검출로 나온다.
_WINS = "tests/test_chat_service.py::test_지정한_범위가_자동_범위를_이긴다"
TARGETS = {
    "지정을 무시하고 늘 자동 범위를 씀": _WINS,
    "지정과 자동 범위의 합집합을 씀": _WINS,
}


def run_tests(target: str | None = None) -> bool:
    paths = [target] if target else ["tests/test_api.py", "tests/test_chat_service.py"]
    result = subprocess.run(
        [str(PY), "-m", "pytest", *paths, "-q", "--no-header"],
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
            path.write_text(original.replace(old, new, 1), encoding="utf-8", newline="\n")
            passed = run_tests(TARGETS.get(label))
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
