"""프롬프트를 고치면서 버전 번호를 안 올리는 것을 막는다 (#61).

## 왜 이 테스트가 있나

버전 번호의 용도는 하나다. **답이 달라졌을 때 프롬프트가 바뀐 탓인지 가리는 것.**
평가 결과(`data/evals/*.json`)와 프로브 기록에 이 번호가 박혀 들어간다.

번호가 올라가지 않은 채 템플릿이 바뀌면, 같은 번호를 달고 다른 프롬프트로 낸
수치 두 개가 저장소에 남는다. 그 둘을 비교하면 틀린 결론이 나오고, **틀렸다는
사실조차 알 수 없다.** 번호가 같으니까.

2026-09-26에 비슷한 사고가 이미 있었다. `settings.yaml`에 `prompt.version: "v1"`이
있었는데 아무도 읽지 않는 값이었고, 실제로 쓰이는 상수는 `v2`였다. 재현성 추적용
설정값이 현실과 다른 값을 담은 채 방치돼 있었다.

## 고치는 법

템플릿을 일부러 바꿨다면 `PROMPT_VERSION`을 올리고 아래 표에 새 줄을 더한다.
표에서 옛 줄을 지우지 않는다. 어느 버전이 어떤 프롬프트였는지가 기록이 된다.
"""

from __future__ import annotations

import hashlib

from techdoc_rag.query.chat_service import _PROMPT_TEMPLATE, PROMPT_VERSION

# 버전 -> 그 버전의 템플릿 sha256. 지우지 말고 더한다.
_TEMPLATE_HASHES = {
    "v2": "02575eb91988ce709d4f6080c5f954d2550718461505517c3991821b2a0b0e37",
}


def test_템플릿을_고치면_버전을_올려야_한다() -> None:
    actual = hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    recorded = _TEMPLATE_HASHES.get(PROMPT_VERSION)

    assert recorded is not None, (
        f"PROMPT_VERSION이 {PROMPT_VERSION}인데 이 테스트의 표에 없습니다. "
        f"버전을 올렸다면 표에 다음 줄을 더하세요.\n"
        f'    "{PROMPT_VERSION}": "{actual}",'
    )
    assert actual == recorded, (
        f"프롬프트 템플릿이 바뀌었는데 PROMPT_VERSION이 {PROMPT_VERSION} 그대로입니다.\n"
        f"  표에 적힌 값: {recorded}\n"
        f"  지금 값:      {actual}\n"
        f"번호를 올리고 표에 새 줄을 더하세요. 번호를 그대로 두면 같은 번호를 달고 "
        f"다른 프롬프트로 낸 평가 결과가 섞입니다."
    )


def test_버전_표에_빈_값이_없다() -> None:
    """표가 비면 위 테스트가 통째로 무력해진다."""
    assert _TEMPLATE_HASHES
    for version, digest in _TEMPLATE_HASHES.items():
        assert version, "버전 이름이 비었습니다"
        assert len(digest) == 64, f"{version}의 sha256 길이가 64가 아닙니다"
