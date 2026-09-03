"""검색된 청크를 프롬프트용 근거 목록으로 조립한다 (#19의 조립 단계).

번호를 붙여 나열하고, 문자 수 예산을 넘으면 관련도(점수)가 낮은 것부터
통째로 뺀다. **청크를 중간에서 자르지 않는다** — 자르면 [번호]가 가리키는
내용과 Citation의 페이지 범위가 어긋난다.

예산 단위가 토큰이 아니라 문자인 것은 청킹과 같은 이유다(settings.yaml
chunking 주석): 토큰을 세려면 토크나이저가 필요한데 임베딩·LLM 모델이
확정(M9) 전이라 지금 토큰 약속은 지킬 수 없다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from techdoc_rag.domain.chunk import RetrievedChunk
from techdoc_rag.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ContextSource:
    """프롬프트에 [number]로 들어간 근거 하나. 답변의 [번호]를 여기로 되짚는다."""

    number: int
    retrieved: RetrievedChunk


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    sources: list[ContextSource]


class ContextBuilder:
    def __init__(self, budget_chars: int) -> None:
        self._budget_chars = budget_chars

    def build(
        self,
        results: list[RetrievedChunk],
        display_names: Mapping[str, str],
    ) -> BuiltContext:
        """점수 내림차순 입력을 전제로, 예산 안에 드는 것만 번호 붙여 조립한다.

        같은 chunk_id 중복은 첫 것만 남긴다. display_names는 document_id →
        사람이 읽는 문서명이며, 없는 문서는 document_id를 그대로 쓴다.
        """
        selected: list[RetrievedChunk] = []
        seen: set[str] = set()
        used_chars = 0
        for result in results:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            entry_length = len(result.chunk.text) + _HEADER_ALLOWANCE
            if used_chars + entry_length > self._budget_chars:
                continue  # 자르지 않고 통째로 뺀다. 더 낮은 점수는 더 클 수 있어 계속 본다
            selected.append(result)
            used_chars += entry_length

        if results and not selected:
            # 청크 하나도 예산에 안 들어가는 것은 관련도 문제가 아니라 설정 문제다.
            # 조용히 빈 근거로 돌리면 No-answer로 둔갑해 원인을 못 찾는다.
            raise ConfigurationError(
                f"근거 예산({self._budget_chars}자)이 청크 하나보다 작음. "
                f"최소 청크 {min(len(r.chunk.text) for r in results)}자"
            )

        sources = [
            ContextSource(number=index + 1, retrieved=result)
            for index, result in enumerate(selected)
        ]
        blocks = []
        for source in sources:
            chunk = source.retrieved.chunk
            name = display_names.get(chunk.document_id, chunk.document_id)
            pages = (
                f"p.{chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"p.{chunk.page_start}~{chunk.page_end}"
            )
            blocks.append(f"[{source.number}] ({name} {pages})\n{chunk.text}")
        return BuiltContext(text="\n\n".join(blocks), sources=sources)


# 번호·문서명·페이지 헤더 몫으로 잡아 두는 문자 수. 정확할 필요는 없고,
# 예산 판정이 본문 길이만 세서 헤더만큼 초과하는 것을 막기 위한 여유분이다.
_HEADER_ALLOWANCE = 80
