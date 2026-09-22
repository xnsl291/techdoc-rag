"""질문에서 제품명을 찾아 검색 범위를 좁히는 부분 테스트 (#52).

저장소는 가짜다. 여기서 보는 것은 이름을 어디서 뽑는가, 어떤 모양일 때
낱말로 인정하는가, 못 찾았을 때 None이 나오는가 세 가지다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from techdoc_rag.domain.document import Document, DocumentStatus
from techdoc_rag.query.document_scope import aliases_by_document, scope_from_question


def _document(document_id: str, logical_document_id: str, filename: str) -> Document:
    return Document(
        document_id=document_id,
        logical_document_id=logical_document_id,
        document_version=1,
        original_filename=filename,
        sha256="a" * 64,
        mime_type="application/pdf",
        file_size_bytes=1,
        page_count=270,
        document_type="manual",
        status=DocumentStatus.READY,
        is_active=True,
        created_at=datetime.now(UTC),
    )


class FakeRepository:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = {item.document_id: item for item in documents}

    def active_document_ids(self) -> list[str]:
        return list(self._documents)

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)


# 실물 6권을 그대로 옮겼다. 4권의 파일 이름이 original.pdf인 것까지 실물이다.
REAL = FakeRepository(
    [
        _document("ls-m100-v1", "ls-m100", "M100_사용설명서_KR_V2.2_251114.pdf"),
        _document("ls-g100-v1", "ls-g100", "G100(C)_사용설명서_KR_V4.03_250829.pdf"),
        _document("ls-h100-v1", "ls-h100", "original.pdf"),
        _document("ls-s100-v1", "ls-s100", "original.pdf"),
        _document("ls-s300-v1", "ls-s300", "original.pdf"),
        _document("ls-is7-v1", "ls-is7", "original.pdf"),
    ]
)


def test_이름은_논리_문서_id의_끝_조각에서_나온다() -> None:
    """파일 이름을 쓰면 안 된다. 6권 중 4권이 original.pdf라 한 이름을 공유한다."""
    assert aliases_by_document(REAL) == {
        "ls-m100-v1": "m100",
        "ls-g100-v1": "g100",
        "ls-h100-v1": "h100",
        "ls-s100-v1": "s100",
        "ls-s300-v1": "s300",
        "ls-is7-v1": "is7",
    }


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # 한글이 바로 붙는 형태. 파이썬 str.isalnum()은 한글에 True를 돌려주므로
        # 경계를 isalnum()으로 보면 이 형태가 전부 빠진다. 실제 평가 문항이다.
        ("S100에서 IO Board Trip은 몇 초 이상 지속되면 발생하나요?", ["ls-s100-v1"]),
        ("M100을 오래 안 쓰고 창고에 둘 건데 몇 도에서 보관해야 하나요?", ["ls-m100-v1"]),
        # 대소문자를 가리지 않는다
        ("iS7 파라미터 초기화 코드는 무엇인가요?", ["ls-is7-v1"]),
        ("is7 제동 저항 연속 사용 시간", ["ls-is7-v1"]),
        # 두 제품을 한 문장에서 묻는 경우
        ("G100과 M100의 주위 온도 조건이 어떻게 다른가요?", ["ls-g100-v1", "ls-m100-v1"]),
    ],
)
def test_이름이_낱말로_서_있으면_그_문서로_좁힌다(question: str, expected: list[str]) -> None:
    assert sorted(scope_from_question(question, aliases_by_document(REAL))) == sorted(expected)


def test_제품명이_없으면_None이다() -> None:
    """빈 목록이 아니라 None이어야 한다. Retriever는 빈 목록을 "이 문서들로
    좁혀라"로 읽어 검색 대상을 0건으로 만든다. 못 찾은 것은 좁히지 말라는 뜻이다."""
    question = "인버터를 다루려면 어떤 자격증이 필요한가요?"

    assert scope_from_question(question, aliases_by_document(REAL)) is None


def test_영숫자에_둘러싸인_이름은_세지_않는다() -> None:
    """LSLV0075M100은 실제로 M100 제품이라 의미상 걸리는 편이 맞지만, 평가셋에서
    두 규칙이 갈리는 문항이 하나뿐이고 답변 불가라 데이터로 못 가렸다.
    잘못 좁히는 쪽이 더 비싸서 보수적인 쪽을 택한 결과다."""
    question = "LSLV0075M100 모델의 가격이 얼마인가요?"

    assert scope_from_question(question, aliases_by_document(REAL)) is None


def test_비활성_문서의_이름은_안_본다() -> None:
    """활성 목록에 없는 문서로 좁히면 검색 대상이 0건이 되어 답이 사라진다."""
    only_m100 = FakeRepository([_document("ls-m100-v1", "ls-m100", "M100.pdf")])

    assert scope_from_question("G100 정기 점검 주기는?", aliases_by_document(only_m100)) is None
