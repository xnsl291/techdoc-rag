"""질의 확장 테스트 (#32).

핵심 둘. 짧은 질문에만 LLM을 부르는 것, 그리고 **확장이 실패해도 원문으로는
검색되는 것** — 보조 기능 하나 때문에 검색 전체가 멈추면 안 된다.
"""

from __future__ import annotations

from collections.abc import Iterator

from techdoc_rag.domain.errors import GenerationError
from techdoc_rag.query.query_expander import QueryExpander


class FakeLlm:
    model_name = "fake"

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]:
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        yield self._reply


def _expander(llm: FakeLlm, short_query_chars: int = 20, max_variants: int = 3) -> QueryExpander:
    return QueryExpander(
        llm_client=llm, short_query_chars=short_query_chars, max_variants=max_variants
    )


def test_짧은_질문은_표현이_늘어난다() -> None:
    llm = FakeLlm("정격 출력 용량 사양\n인버터 출력 kW 정격")

    variants = _expander(llm).expand("정격출력")

    assert variants[0] == "정격출력"  # 원문이 항상 첫 번째
    assert "정격 출력 용량 사양" in variants
    assert len(variants) == 3


def test_긴_질문은_LLM을_부르지_않는다() -> None:
    """긴 질문은 이미 맥락이 충분하다. 부르면 응답만 느려진다."""
    llm = FakeLlm("불려서는 안 됨")

    variants = _expander(llm).expand("인버터를 설치할 때 주위 온도 조건이 어떻게 되는지 알려주세요")

    assert variants == ["인버터를 설치할 때 주위 온도 조건이 어떻게 되는지 알려주세요"]
    assert llm.prompts == []


def test_확장이_실패해도_원문으로는_검색한다() -> None:
    """LLM이 죽어도 검색은 돼야 한다. 확장은 전제 조건이 아니라 보조 기능이다."""
    llm = FakeLlm(error=GenerationError("LLM 서버 접속 실패"))

    assert _expander(llm).expand("정격출력") == ["정격출력"]


def test_번호와_기호가_붙어_와도_떼어낸다() -> None:
    """지시를 무시하고 목록 기호를 붙이는 경우가 있다."""
    llm = FakeLlm("1. 정격 출력 사양\n- 출력 용량 kW\n* 정격 파워")

    variants = _expander(llm, max_variants=4).expand("정격출력")

    assert "정격 출력 사양" in variants
    assert "출력 용량 kW" in variants
    assert not any(v.startswith(("-", "*", "1.")) for v in variants)


def test_중복과_빈_줄은_버린다() -> None:
    llm = FakeLlm("정격출력\n\n정격 출력 사양\n정격 출력 사양\n")

    variants = _expander(llm, max_variants=5).expand("정격출력")

    assert variants == ["정격출력", "정격 출력 사양"]


def test_상한을_넘겨_받아도_상한까지만_쓴다() -> None:
    llm = FakeLlm("표현1\n표현2\n표현3\n표현4\n표현5")

    assert len(_expander(llm, max_variants=3).expand("정격출력")) == 3
