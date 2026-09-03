"""설정 로딩.

비밀이 아닌 값은 config/settings.yaml에서, 엔드포인트와 API 키는 .env에서 읽는다.
settings.yaml은 커밋되고 .env는 커밋되지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from techdoc_rag.domain.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True, slots=True)
class LlmSettings:
    model_name: str
    temperature: float
    thinking_enabled: bool
    runtime_context_tokens: int
    generation_max_tokens: int
    endpoint: str
    max_concurrent_generations: int
    queue_timeout_seconds: int
    generation_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model_name: str
    max_input_tokens: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class IndexingSettings:
    heartbeat_interval_seconds: int
    lease_seconds: int


@dataclass(frozen=True, slots=True)
class ChunkingSettings:
    size_chars: int
    overlap_chars: int
    config_version: str


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    top_k: int
    similarity_threshold: float
    context_budget_chars: int


@dataclass(frozen=True, slots=True)
class ApiSettings:
    max_question_chars: int


@dataclass(frozen=True, slots=True)
class StorageSettings:
    document_root: Path
    metadata_database_path: Path
    qdrant_endpoint: str
    qdrant_api_key: str | None
    qdrant_collection_name: str
    qdrant_read_alias: str


@dataclass(frozen=True, slots=True)
class Settings:
    llm: LlmSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    indexing: IndexingSettings
    retrieval: RetrievalSettings
    api: ApiSettings
    storage: StorageSettings
    prompt_version: str


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigurationError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _require(section: dict, key: str, section_name: str):
    if key not in section:
        raise ConfigurationError(f"settings.yaml의 {section_name}에 {key}가 없습니다")
    return section[key]


def load_settings(settings_path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """settings.yaml과 .env를 읽어 하나의 Settings로 만든다.

    값이 비어 있으면 기본값으로 넘어가지 않고 바로 실패시킨다.
    설정 누락이 조용히 기본값으로 대체되면 어떤 설정으로 낸 결과인지 알 수 없게 되고,
    재현성(NFR-004)이 깨진다.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    raw = _read_yaml(settings_path)

    llm_section = raw.get("llm", {})
    llm = LlmSettings(
        model_name=_require(llm_section, "model_name", "llm"),
        temperature=float(_require(llm_section, "temperature", "llm")),
        thinking_enabled=bool(_require(llm_section, "thinking_enabled", "llm")),
        runtime_context_tokens=int(_require(llm_section, "runtime_context_tokens", "llm")),
        generation_max_tokens=int(_require(llm_section, "generation_max_tokens", "llm")),
        endpoint="http://{host}:{port}".format(
            host=os.getenv("OLLAMA_HOST", "127.0.0.1"),
            port=os.getenv("OLLAMA_PORT", "11434"),
        ),
        max_concurrent_generations=int(
            _require(llm_section, "max_concurrent_generations", "llm")
        ),
        queue_timeout_seconds=int(_require(llm_section, "queue_timeout_seconds", "llm")),
        generation_timeout_seconds=int(
            _require(llm_section, "generation_timeout_seconds", "llm")
        ),
    )

    embedding_section = raw.get("embedding", {})
    indexing_section = raw.get("indexing", {})
    chunking_section = raw.get("chunking", {})
    retrieval_section = raw.get("retrieval", {})
    storage_section = raw.get("storage", {})

    return Settings(
        llm=llm,
        embedding=EmbeddingSettings(
            model_name=_require(embedding_section, "model_name", "embedding"),
            max_input_tokens=int(_require(embedding_section, "max_input_tokens", "embedding")),
            batch_size=int(_require(embedding_section, "batch_size", "embedding")),
        ),
        indexing=IndexingSettings(
            heartbeat_interval_seconds=int(
                _require(indexing_section, "heartbeat_interval_seconds", "indexing")
            ),
            lease_seconds=int(_require(indexing_section, "lease_seconds", "indexing")),
        ),
        chunking=ChunkingSettings(
            size_chars=int(_require(chunking_section, "size_chars", "chunking")),
            overlap_chars=int(_require(chunking_section, "overlap_chars", "chunking")),
            config_version=_require(chunking_section, "config_version", "chunking"),
        ),
        retrieval=RetrievalSettings(
            top_k=int(_require(retrieval_section, "top_k", "retrieval")),
            similarity_threshold=float(
                _require(retrieval_section, "similarity_threshold", "retrieval")
            ),
            context_budget_chars=int(
                _require(retrieval_section, "context_budget_chars", "retrieval")
            ),
        ),
        api=ApiSettings(
            max_question_chars=int(
                _require(raw.get("api", {}), "max_question_chars", "api")
            ),
        ),
        storage=StorageSettings(
            document_root=PROJECT_ROOT / _require(storage_section, "document_root", "storage"),
            metadata_database_path=PROJECT_ROOT
            / _require(storage_section, "metadata_database_path", "storage"),
            qdrant_endpoint="http://{host}:{port}".format(
                host=os.getenv("QDRANT_HOST", "127.0.0.1"),
                port=os.getenv("QDRANT_PORT", "6333"),
            ),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection_name=_require(
                storage_section, "qdrant_collection_name", "storage"
            ),
            qdrant_read_alias=_require(storage_section, "qdrant_read_alias", "storage"),
        ),
        prompt_version=raw.get("prompt", {}).get("version", "v1"),
    )
