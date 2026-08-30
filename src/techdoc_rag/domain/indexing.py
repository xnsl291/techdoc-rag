"""한 번의 색인 실행.

청크마다 같은 값이라 Chunk에 넣지 않고 따로 둔다. Chunk는 문서를 쪼갠 조각이고,
문서가 어느 계열에 속하는지나 어떤 종류인지는 문서 수준 정보다.

이 값들이 벡터 payload에 복제되는 이유는 검색 필터 때문이다. 계열이나 종류로
좁히려면 벡터 저장소가 그 값을 알아야 하고, 모르면 검색 결과마다 문서 장부를
다시 조회해야 한다.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

# 문서 계열 ID 형식. {제조사}-{모델명} 소문자.
# 느슨하게만 검사한다. 조이면 나중에 예외가 생겼을 때 규칙부터 고쳐야 한다.
LOGICAL_DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)+$")


def new_index_run_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class IndexRun:
    """색인 실행 한 번을 식별하고, 그 실행이 어떤 문서를 다루는지 담는다.

    index_run_id는 재색인 시 남는 벡터를 지우는 근거다. 청크 수가 30개에서
    22개로 줄면 이전 8개가 그대로 남는데, 같은 document_id라 활성 필터를
    통과한다. 색인이 끝난 뒤 이번 실행의 ID가 아닌 벡터를 지우면 정리된다.
    """

    logical_document_id: str
    document_type: str
    index_run_id: str = field(default_factory=new_index_run_id)

    def __post_init__(self) -> None:
        if not LOGICAL_DOCUMENT_ID_PATTERN.match(self.logical_document_id):
            raise ValueError(
                f"문서 계열 ID 형식이 맞지 않음: {self.logical_document_id!r}. "
                f"소문자와 숫자를 하이픈으로 이은 형태여야 함 (예: ls-g100)"
            )
        if not self.document_type:
            raise ValueError("document_type이 비어 있음")
