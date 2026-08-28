"""Recursive 청커 (DP-09 Baseline).

외부 라이브러리를 쓰지 않는 순수 파이썬이라 adapters가 아니라 ingestion에 둔다.
교체 가능성(Section-aware 개선)은 domain.ports.Chunker Protocol이 보장한다.

크기 단위는 문자 수다. 토큰을 세려면 토크나이저가 필요한데 임베딩 모델이
미정(M9)이므로, 지키지 못하는 약속 대신 문자 수로 정직하게 시작한다.
"""

from __future__ import annotations

from bisect import bisect_right

from techdoc_rag.domain.chunk import Chunk
from techdoc_rag.domain.parsing import ParsedDocument

#: 큰 단위부터 시도한다. 문단이 안 끊기게 최대한 크게, 안 되면 한 단계 작게.
#: 마지막 빈 문자열은 "구분자가 없으면 그냥 크기대로 자른다"는 뜻이다.
_SEPARATORS = ("\n\n", "\n", " ", "")


def _page_boundaries(parsed_document: ParsedDocument) -> tuple[str, list[int], list[int]]:
    """페이지들을 하나의 텍스트로 잇고, 각 페이지의 시작 오프셋을 기록한다.

    청킹 로직과 분리해 둔 것은 페이지 매핑이 청킹과 독립적으로 틀릴 수 있는
    부분이기 때문이다. 여기가 틀리면 Citation이 엉뚱한 페이지를 가리킨다.
    """
    parts: list[str] = []
    page_start_offsets: list[int] = []
    page_numbers: list[int] = []
    offset = 0
    for page in parsed_document.pages:
        page_start_offsets.append(offset)
        page_numbers.append(page.page_no)
        # 페이지 사이에 개행을 두어 마지막 단어와 다음 페이지 첫 단어가 붙지 않게 한다.
        parts.append(page.text + "\n")
        offset += len(page.text) + 1
    return "".join(parts), page_start_offsets, page_numbers


class RecursiveChunker:
    """구분자 우선순위에 따라 재귀적으로 나누고, 목표 크기로 묶는다."""

    def __init__(self, size_chars: int, overlap_chars: int, config_version: str) -> None:
        if size_chars <= 0:
            raise ValueError("size_chars는 양수여야 함")
        if not 0 <= overlap_chars < size_chars:
            raise ValueError("overlap_chars는 0 이상 size_chars 미만이어야 함")
        self._size_chars = size_chars
        self._overlap_chars = overlap_chars
        self._config_version = config_version

    @property
    def chunk_config_version(self) -> str:
        return self._config_version

    def chunk(
        self,
        parsed_document: ParsedDocument,
        document_id: str,
        document_version: int,
    ) -> list[Chunk]:
        full_text, page_start_offsets, page_numbers = _page_boundaries(parsed_document)
        if not full_text.strip():
            return []

        segment_spans = self._split_span(full_text, 0, len(full_text), 0)
        chunk_spans = self._pack_with_overlap(segment_spans)

        def page_of(offset: int) -> int:
            return page_numbers[bisect_right(page_start_offsets, offset) - 1]

        chunks: list[Chunk] = []
        for start, end in chunk_spans:
            # 앞뒤 공백은 청크 본문에서만 걷어내고, 페이지 계산은 걷어낸 위치로 한다.
            while start < end and full_text[start].isspace():
                start += 1
            while end > start and full_text[end - 1].isspace():
                end -= 1
            if start == end:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{len(chunks):04d}",
                    document_id=document_id,
                    document_version=document_version,
                    page_start=page_of(start),
                    page_end=page_of(end - 1),
                    text=full_text[start:end],
                )
            )
        return chunks

    def _split_span(
        self, text: str, start: int, end: int, separator_index: int
    ) -> list[tuple[int, int]]:
        """[start, end) 구간을 size_chars 이하 조각들로 나눈다.

        현재 구분자로 나눠 본 뒤, 여전히 큰 조각만 다음 구분자로 재귀한다.
        """
        if end - start <= self._size_chars:
            return [(start, end)]

        separator = _SEPARATORS[separator_index]
        if separator == "":
            # 구분자가 바닥났다. 크기대로 자르는 것 외에 방법이 없다.
            return [
                (position, min(position + self._size_chars, end))
                for position in range(start, end, self._size_chars)
            ]

        spans: list[tuple[int, int]] = []
        piece_start = start
        while piece_start < end:
            found = text.find(separator, piece_start, end)
            piece_end = end if found == -1 else found + len(separator)
            if piece_end - piece_start > self._size_chars:
                spans.extend(self._split_span(text, piece_start, piece_end, separator_index + 1))
            else:
                spans.append((piece_start, piece_end))
            piece_start = piece_end
        return spans

    def _pack_with_overlap(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """작은 조각들을 목표 크기까지 묶고, 청크 사이에 겹침을 둔다.

        겹침은 앞 청크의 끝부분을 다음 청크 앞에 붙이는 것이다. 문장이 청크
        경계에서 잘려 의미를 잃는 것을 줄인다.

        크기 보장은 size_chars + overlap_chars다. 겹침으로 시작한 청크에
        size_chars에 가까운 조각이 바로 이어지는 경우가 상한이다.
        """
        if not spans:
            return []
        chunks: list[tuple[int, int]] = []
        current_start, current_end = spans[0]
        for _span_start, span_end in spans[1:]:
            if span_end - current_start <= self._size_chars:
                current_end = span_end
                continue
            chunks.append((current_start, current_end))
            current_start = max(current_start + 1, current_end - self._overlap_chars)
            current_end = span_end
        chunks.append((current_start, current_end))
        return chunks
