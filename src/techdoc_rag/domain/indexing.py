"""한 번의 색인 실행.

청크마다 같은 값이라 Chunk에 넣지 않고 따로 둔다. Chunk는 문서를 쪼갠 조각이고,
문서가 어느 계열에 속하는지나 어떤 종류인지는 문서 수준 정보다.

이 값들이 벡터 payload에 복제되는 이유는 검색 필터 때문이다. 계열이나 종류로
좁히려면 벡터 저장소가 그 값을 알아야 하고, 모르면 검색 결과마다 SQLite를
다시 조회해야 한다.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

# 문서 계열 ID 형식. {제조사}-{모델명} 소문자.
# 느슨하게만 검사한다. 엄격하게 하면 나중에 예외가 생겼을 때 규칙부터 고쳐야 한다.
LOGICAL_DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)+$")


def new_index_run_id() -> str:
    return uuid.uuid4().hex


def validate_logical_document_id(value: str) -> None:
    """문서 계열 ID 형식을 검사한다.

    검사 지점을 하나로 둔다. 등록과 색인이 각자 검사하면 규칙이 갈라진다.
    """
    if not LOGICAL_DOCUMENT_ID_PATTERN.match(value):
        raise ValueError(
            f"문서 계열 ID 형식이 맞지 않음: {value!r}. "
            f"소문자와 숫자를 하이픈으로 이은 형태여야 함 (예: ls-g100)"
        )


@dataclass(frozen=True, slots=True)
class IndexRun:
    """색인 실행 한 번을 식별하고, 그 실행이 어떤 문서를 다루는지 담는다.

    index_run_id는 재색인 시 남는 벡터를 지우는 근거다. 청크 수가 30개에서
    22개로 줄면 이전 8개가 그대로 남는데, 같은 document_id라 활성 필터를
    통과한다. 색인이 끝난 뒤 이번 실행의 ID가 아닌 벡터를 지우면 정리된다.

    index_run_id에 기본값을 두지 않는다. 한 색인에서 IndexRun을 두 번 만들면
    ID가 달라져 앞서 저장한 벡터가 통째로 고아가 된다. 호출자가 명시하게 한다.

    document_id를 함께 들고 있는 이유는 두 가지다. upsert가 청크와 이 값이
    같은 문서인지 검사할 수 있고, 고아 벡터를 지울 때 필요한 조건
    (document_id = X AND index_run_id != 이번 실행)이 여기서 모두 나온다.
    """

    document_id: str
    logical_document_id: str
    document_type: str
    index_run_id: str

    def __post_init__(self) -> None:
        validate_logical_document_id(self.logical_document_id)
        if not self.document_type:
            raise ValueError("document_type이 비어 있음")
        if not self.index_run_id:
            raise ValueError("index_run_id가 비어 있음")
