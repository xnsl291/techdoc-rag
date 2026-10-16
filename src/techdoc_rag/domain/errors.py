"""단계별 실패 구분.

Parsing, Retrieval, Generation 실패를 하나로 뭉치지 않는다.
검색이 실패했을 때 LLM의 일반 지식으로 우회하면 근거 없는 답이 나가고,
로그에는 성공으로 남아 원인을 추적할 수 없게 된다.
"""

from __future__ import annotations


class TechdocRagError(Exception):
    """이 시스템이 발생시키는 모든 예외의 최상위."""


class ConfigurationError(TechdocRagError):
    """설정값이 없거나 잘못된 경우."""


class ParsingError(TechdocRagError):
    """PDF에서 텍스트를 추출하지 못한 경우."""


class IndexingError(TechdocRagError):
    """임베딩 생성 또는 벡터 저장에 실패한 경우."""


class MetadataStoreError(TechdocRagError):
    """문서 메타데이터 저장소 접근 또는 제약 위반.

    중복 등록, 활성 전환 실패, DB 연결 실패가 여기로 온다.
    Parsing/Retrieval/Generation과 구분되는 네 번째 실패 축이다.
    """


class RetrievalError(TechdocRagError):
    """벡터 저장소 조회에 실패한 경우.

    검색 결과가 비어 있는 것은 실패가 아니라 No-answer로 처리한다.
    이 예외는 저장소에 접근하지 못했을 때만 쓴다.
    """


class GenerationError(TechdocRagError):
    """LLM 호출 또는 응답 생성에 실패한 경우."""


class UnknownDocumentError(TechdocRagError):
    """질의 범위로 지정한 문서가 없거나 비활성인 경우(#43).

    장애가 아니라 요청이 틀린 것이다. 조용히 걸러 내면 범위가 0건이 되어
    "관련 내용을 찾지 못했습니다"로 끝나는데, 그러면 사용자는 매뉴얼에 답이
    없다고 오해한다. 실제로는 문서 ID를 잘못 준 것이다.
    """

    def __init__(self, document_ids: list[str]) -> None:
        self.document_ids = document_ids
        super().__init__(f"없거나 비활성인 문서: {', '.join(document_ids)}")
