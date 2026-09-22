"""질문에 적힌 제품명으로 검색 범위를 좁힌다 (#52의 A부류).

문서 6권이 전부 같은 회사의 인버터 매뉴얼이라 문장이 서로 닮았다. "팬 교체
주기"는 여섯 권 모두에 있고, 질문이 S100을 가리켜도 H100 쪽 문장이 더 가까우면
S100이 8~9위로 밀려난다. 2026-09-17 실측에서 실패 13문항 중 6건이 이 부류였다.

질문에 제품명이 적혀 있으면 그 문서로만 검색한다. 제품명을 못 찾으면 아무것도
넘기지 않는다. 어설프게 좁히면 지금보다 나빠지기 때문이다.

`agent/tools.py`의 `_resolve`와 겹쳐 보이지만 문제가 다르다. 저쪽은 모델이
"G100"이라고 **골라서 건네준 이름**을 id로 바꾸는 일이라 부분 문자열 대조로
충분하다. 여기는 자유 문장을 훑어 이름을 **찾아내는** 일이라 잘못 걸리는 쪽이
위험하다. 그래서 경계 검사가 붙는다.

`query/`에 두는 이유는 `agent/`가 `query/`를 쓰기 때문이다. 반대로 두면 서로를
부르는 꼴이 된다(DP-43).
"""

from __future__ import annotations

import re

from techdoc_rag.domain.ports import DocumentRepository


def aliases_by_document(repository: DocumentRepository) -> dict[str, str]:
    """활성 문서마다 질문에서 찾아볼 이름을 하나씩 모은다.

    `logical_document_id`의 끝 조각을 쓴다(`ls-m100` -> `m100`).

    파일 이름은 쓰지 않는다. 수집 스크립트가 원본을 `<폴더>/original.pdf`로
    저장해서 6권 중 4권의 `original_filename`이 전부 `original.pdf`다. 그걸
    이름으로 쓰면 네 문서가 한 이름을 공유해 범위가 안 좁혀진다.

    `document_id`에서 `-v1`을 떼는 방법도 되지만, 그러면 id 형식을 코드가
    추측하게 된다. 저장소가 `logical_document_id`를 따로 갖고 있으니 그걸 읽는다.
    """
    aliases: dict[str, str] = {}
    for document_id in repository.active_document_ids():
        document = repository.get(document_id)
        if document is None:
            continue
        alias = document.logical_document_id.rsplit("-", 1)[-1]
        if alias:
            aliases[document_id] = alias
    return aliases


def scope_from_question(question: str, aliases: dict[str, str]) -> list[str] | None:
    """질문에 이름이 박힌 문서들의 id를 돌려준다. 하나도 없으면 None.

    None과 빈 목록은 뜻이 다르다. `Retriever.retrieve`는 빈 목록을 "이 문서들로
    좁혀라"로 읽어 검색 대상을 0건으로 만든다. 제품명을 못 찾은 것은 좁히지
    말라는 뜻이므로 None이어야 한다.

    이름이 여럿 걸리면 전부 넘긴다. "G100과 M100 중 어느 쪽이"처럼 두 제품을
    한 문장에서 묻는 경우가 있다.
    """
    matched = [
        document_id
        for document_id, alias in aliases.items()
        if _mentions(question, alias)
    ]
    return matched or None


def _mentions(question: str, alias: str) -> bool:
    """이름이 낱말로 서 있는지 본다.

    경계를 ASCII 영숫자로만 판정하는 것이 핵심이다. 파이썬 `str.isalnum()`은
    한글에도 True를 돌려주므로, 그걸로 경계를 보면 "S100에서 IO Board Trip은"의
    `에` 때문에 매칭이 전부 빠진다.

    경계를 요구하면 `LSLV0075M100` 같은 모델 코드 안의 `M100`은 안 걸린다. 그
    코드는 실제로 M100 제품이라 의미상으로는 걸리는 편이 맞지만, 평가셋 30문항
    중 두 규칙이 갈리는 문항이 q010 하나뿐이고 그마저 답변 불가 문항이라
    데이터로 가릴 수 없었다. 못 가르는 자리에서는 잘못 좁히는 쪽이 더 비싸므로
    보수적인 쪽을 택했다. 모델 코드로 묻는 실제 질문이 모이면 다시 본다.
    """
    pattern = rf"(?<![0-9A-Za-z]){re.escape(alias)}(?![0-9A-Za-z])"
    return re.search(pattern, question, re.IGNORECASE) is not None
