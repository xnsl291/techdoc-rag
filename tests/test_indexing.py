"""IndexRun 검증 테스트.

이 값들이 잘못되면 벡터 payload에 사실과 다른 정보가 박히는데, 오류가 나지 않아
검색 결과가 조용히 틀린다. 그래서 만드는 시점에 막는다.
"""

from __future__ import annotations

import pytest

from techdoc_rag.domain.indexing import IndexRun, new_index_run_id, validate_logical_document_id


def _index_run(**overrides: str) -> IndexRun:
    values = {
        "document_id": "ls-1828",
        "logical_document_id": "ls-g100",
        "document_type": "manual",
        "index_run_id": "run-0001",
    }
    values.update(overrides)
    return IndexRun(**values)


def test_정상적인_값은_통과한다() -> None:
    run = _index_run()

    assert run.document_id == "ls-1828"
    assert run.logical_document_id == "ls-g100"


@pytest.mark.parametrize(
    "invalid",
    ["LS-G100", "ls_g100", "g100", "ls-G100", "ls g100", "", "ls-"],
)
def test_계열_ID_형식이_틀리면_거부한다(invalid: str) -> None:
    with pytest.raises(ValueError, match="문서 계열 ID"):
        _index_run(logical_document_id=invalid)


def test_document_type이_비면_거부한다() -> None:
    with pytest.raises(ValueError, match="document_type"):
        _index_run(document_type="")


def test_index_run_id가_비면_거부한다() -> None:
    with pytest.raises(ValueError, match="index_run_id"):
        _index_run(index_run_id="")


def test_새_실행_ID는_매번_다르다() -> None:
    assert new_index_run_id() != new_index_run_id()


def test_검증_함수를_직접_쓸_수_있다() -> None:
    """등록과 색인이 같은 함수를 쓴다. 각자 검사하면 규칙이 갈라진다."""
    validate_logical_document_id("ls-g100")

    with pytest.raises(ValueError):
        validate_logical_document_id("manual-g100-UPPER")
