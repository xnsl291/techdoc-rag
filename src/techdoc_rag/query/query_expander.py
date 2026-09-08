"""짧은 질문을 검색용 표현 여러 개로 넓힌다 (#32, DP-17).

실물에서 "정격출력"처럼 짧은 용어형 질문이 표현에 따라 서로 다른 근거를 물어
왔다. 같은 뜻 6가지로 물었을 때 상위 3건 중 겹치는 것이 0~2건이었다
(2026-09-07 측정). 한 번만 검색하면 그중 어느 하나에 걸리는 것이 운에 달린다.

여러 표현으로 검색해 결과를 합치면 그 운을 줄인다. 대신 LLM 호출이 한 번
늘어난다. 그래서 **짧은 질문에만** 적용한다 — 긴 질문은 이미 맥락이 충분해
확장의 이득이 적다(수행자 확정 2026-09-07).

**확장 실패가 검색을 막지 않는다.** LLM이 죽어 있어도 원문으로는 검색해야
한다. 확장은 검색을 돕는 보조 기능이지 전제 조건이 아니다.
"""

from __future__ import annotations

import logging

from techdoc_rag.domain.errors import GenerationError
from techdoc_rag.domain.ports import LlmClient

_logger = logging.getLogger(__name__)

_PROMPT = """다음 질문을 기술 매뉴얼에서 검색하기 좋은 표현 {count}개로 바꿔라.

규칙:
- 한 줄에 하나씩, 번호나 기호를 붙이지 마라
- 질문을 설명하지 말고 검색어만 써라
- 매뉴얼에 실제로 쓰일 법한 용어를 포함하라
- 원래 질문의 의도를 바꾸지 마라

질문: {question}
"""


class QueryExpander:
    def __init__(
        self,
        llm_client: LlmClient,
        short_query_chars: int,
        max_variants: int,
        max_tokens: int = 128,
    ) -> None:
        self._llm_client = llm_client
        self._short_query_chars = short_query_chars
        self._max_variants = max_variants
        self._max_tokens = max_tokens

    def expand(self, question: str) -> list[str]:
        """검색에 쓸 표현 목록. 첫 번째는 항상 원문이다.

        원문을 빼면 모델이 질문을 잘못 바꿨을 때 되돌릴 길이 없어진다.
        """
        if len(question.strip()) > self._short_query_chars:
            return [question]

        prompt = _PROMPT.format(count=self._max_variants - 1, question=question)
        try:
            text = "".join(self._llm_client.generate(prompt, max_tokens=self._max_tokens))
        except GenerationError as error:
            # 확장에 실패해도 원문으로는 검색한다. 여기서 예외를 올리면
            # 보조 기능 하나 때문에 검색 전체가 멈춘다.
            _logger.warning("질의 확장 실패, 원문으로 검색함: %s", error)
            return [question]

        variants = [question]
        for line in text.splitlines():
            candidate = line.strip().lstrip("-•*0123456789. )")
            if candidate and candidate not in variants:
                variants.append(candidate)
            if len(variants) >= self._max_variants:
                break
        return variants
