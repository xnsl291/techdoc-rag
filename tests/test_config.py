"""설정 로딩 테스트 (#21).

지금까지 테스트가 하나도 없었다. 설정 키 이름 오타를 잡아줄 것이 없으면
색인이 한참 돌다 엉뚱한 곳에서 터진다 — `load_settings`는 값이 없을 때
기본값으로 넘어가지 않고 바로 실패시키는 것이 계약이다(NFR-004 재현성).

커밋된 config/settings.yaml이 실제로 로드되는지도 함께 본다. 필수 키를
코드에 추가하면서 yaml에 안 넣으면 여기서 잡힌다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from techdoc_rag.config import DEFAULT_SETTINGS_PATH, load_settings
from techdoc_rag.domain.errors import ConfigurationError


def _complete_settings() -> dict:
    """커밋된 설정을 그대로 읽어 온다.

    테스트 안에 값을 다시 적으면 실제 파일과 갈라진다 — 그러면 이 테스트는
    실물이 아니라 자기 사본을 검증하게 된다.
    """
    with DEFAULT_SETTINGS_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write(path: Path, data: dict) -> Path:
    settings_path = path / "settings.yaml"
    with settings_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True)
    return settings_path


def test_커밋된_설정이_로드된다() -> None:
    """실제 파일이 코드가 요구하는 키를 전부 갖고 있는지 본다."""
    settings = load_settings()

    assert settings.llm.model_name
    assert settings.embedding.model_name
    assert settings.chunking.size_chars > 0
    assert settings.retrieval.top_k > 0
    assert settings.retrieval.context_budget_chars > 0
    assert settings.api.max_question_chars > 0
    assert settings.indexing.lease_seconds > 0
    assert settings.storage.qdrant_collection_name
    assert settings.storage.qdrant_read_alias


def test_설정_파일이_없으면_ConfigurationError(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="찾을 수 없"):
        load_settings(tmp_path / "없는파일.yaml")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("llm", "model_name"),
        ("llm", "max_concurrent_generations"),
        ("embedding", "model_name"),
        ("embedding", "batch_size"),
        ("chunking", "size_chars"),
        ("chunking", "config_version"),
        ("indexing", "lease_seconds"),
        ("retrieval", "top_k"),
        ("retrieval", "context_budget_chars"),
        ("api", "max_question_chars"),
        ("storage", "metadata_database_path"),
        ("storage", "qdrant_collection_name"),
        ("storage", "qdrant_read_alias"),
    ],
)
def test_필수_키가_빠지면_어느_키인지_알려준다(
    tmp_path: Path, section: str, key: str
) -> None:
    """조용히 기본값으로 넘어가면 어떤 설정으로 낸 결과인지 알 수 없게 된다."""
    data = _complete_settings()
    del data[section][key]

    with pytest.raises(ConfigurationError) as caught:
        load_settings(_write(tmp_path, data))

    message = str(caught.value)
    assert key in message  # 빠진 키 이름
    assert section in message  # 어느 절인지


def test_섹션_자체가_없어도_ConfigurationError(tmp_path: Path) -> None:
    data = _complete_settings()
    del data["retrieval"]

    with pytest.raises(ConfigurationError, match="top_k"):
        load_settings(_write(tmp_path, data))


def test_빈_파일도_ConfigurationError(tmp_path: Path) -> None:
    """yaml.safe_load가 None을 돌려주는 경로 — 여기서 AttributeError가 나면 안 된다."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_settings(settings_path)


def test_숫자_설정에_문자열이_들어오면_실패한다(tmp_path: Path) -> None:
    """int() 변환이 조용히 통과하면 top_k='다섯' 같은 값이 검색 시점에 터진다."""
    data = _complete_settings()
    data["retrieval"]["top_k"] = "다섯"

    with pytest.raises(ValueError):
        load_settings(_write(tmp_path, data))


def test_저장소_경로가_프로젝트_루트_기준으로_풀린다(tmp_path: Path) -> None:
    """상대 경로를 그대로 쓰면 실행 디렉터리에 따라 다른 DB를 연다."""
    settings = load_settings()

    assert settings.storage.metadata_database_path.is_absolute()
    assert settings.storage.document_root.is_absolute()
