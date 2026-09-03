"""FastAPI 앱 — 파이프라인의 HTTP 경계 (#27).

create_app은 조립이 끝난 의존성을 받는다. 테스트는 가짜를 넣고,
운영은 create_default_app(조립 지점)이 settings.yaml로 실물을 만든다.
서버는 stateless다(DP-44) — 대화 상태를 들고 있지 않는다.

실행:
    uvicorn --factory techdoc_rag.api.app:create_default_app --host 127.0.0.1 --port 8000

127.0.0.1 바인딩은 보안 요구다(05 보안 절) — Ollama·Qdrant와 같은 원칙으로
이 서비스도 외부 인터페이스에 열지 않는다. 인증이 없기 때문이다.
"""

from __future__ import annotations

import http.client
from collections.abc import Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException

from techdoc_rag.api.schemas import ChatRequest, ChatResponse, HealthResponse
from techdoc_rag.domain.errors import GenerationError, RetrievalError
from techdoc_rag.query.chat_service import ChatService

# 이름 → 접근 확인 함수. 실패는 예외로 알린다.
HealthProbes = dict[str, Callable[[], None]]


def create_app(
    chat_service: ChatService,
    health_probes: HealthProbes,
    max_question_chars: int,
) -> FastAPI:
    app = FastAPI(title="techdoc-rag", docs_url=None, redoc_url=None)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if len(request.question) > max_question_chars:
            # 길이 상한은 프롬프트 예산 보호다. 긴 질문을 조용히 자르면
            # 모델이 받은 질문과 사용자가 보낸 질문이 달라진다.
            raise HTTPException(
                status_code=422,
                detail=f"질문이 너무 김: {len(request.question)}자 (상한 {max_question_chars}자)",
            )
        try:
            answer = chat_service.ask(request.question)
        except (RetrievalError, GenerationError) as error:
            # 장애는 No-answer가 아니라 503이다(D-005). LLM 일반 지식으로
            # 우회하지 않았다는 사실이 상태 코드로 드러난다.
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ChatResponse.from_answer(answer)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        components: dict[str, str] = {}
        for name, probe in health_probes.items():
            try:
                probe()
                components[name] = "ok"
            except Exception as error:  # noqa: BLE001 (어떤 실패든 보고가 목적)
                components[name] = f"실패: {error}"
        response = HealthResponse(
            status="ok" if all(value == "ok" for value in components.values()) else "degraded",
            components=components,
        )
        if response.status != "ok":
            # 죽은 구성요소가 있으면 로드밸런서·감시가 상태 코드만 보고도 알게 한다.
            raise HTTPException(status_code=503, detail=response.model_dump())
        return response

    return app


def create_default_app() -> FastAPI:
    """settings.yaml로 실물 어댑터를 조립한다 — 유일한 운영 조립 지점.

    스크립트마다 따로 조립하면 값이 갈린다. 검증 스크립트들도 차차 여기를 쓰게 한다.
    """
    from qdrant_client import QdrantClient

    from techdoc_rag.adapters.ollama_embedding_model import OllamaEmbeddingModel
    from techdoc_rag.adapters.ollama_llm_client import OllamaLlmClient
    from techdoc_rag.adapters.qdrant_vector_store import QdrantVectorStore
    from techdoc_rag.adapters.sqlite_document_repository import SqliteDocumentRepository
    from techdoc_rag.config import load_settings
    from techdoc_rag.query.context_builder import ContextBuilder
    from techdoc_rag.query.retriever import Retriever

    settings = load_settings()
    repository = SqliteDocumentRepository(settings.storage.metadata_database_path)
    repository.initialize()
    # Qdrant는 로컬 파일 모드(DP-52). 서버 전환은 평가 수치 측정 직전이 경계다 —
    # 로컬은 완전 탐색, 서버는 근사 검색이라 순위가 달라 평가는 서버 모드로 재야 한다.
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=str(settings.storage.metadata_database_path.parent / "qdrant")),
        collection_name=settings.storage.qdrant_collection_name,
        vector_size=1024,  # bge-m3 (DP-49)
    )
    vector_store.initialize()
    embedding = OllamaEmbeddingModel(
        model_name=settings.embedding.model_name,
        endpoint=settings.llm.endpoint,
        batch_size=settings.embedding.batch_size,
        num_batch=settings.embedding.max_input_tokens,
        embedding_version="v1",
    )
    llm = OllamaLlmClient(
        model_name=settings.llm.model_name,
        endpoint=settings.llm.endpoint,
        temperature=settings.llm.temperature,
        runtime_context_tokens=settings.llm.runtime_context_tokens,
        thinking_enabled=settings.llm.thinking_enabled,
        max_concurrent_generations=settings.llm.max_concurrent_generations,
        queue_timeout_seconds=settings.llm.queue_timeout_seconds,
        generation_timeout_seconds=settings.llm.generation_timeout_seconds,
    )
    chat_service = ChatService(
        retriever=Retriever(
            embedding_model=embedding,
            vector_store=vector_store,
            repository=repository,
            top_k=settings.retrieval.top_k,
            similarity_threshold=settings.retrieval.similarity_threshold,
        ),
        context_builder=ContextBuilder(budget_chars=settings.retrieval.context_budget_chars),
        llm_client=llm,
        repository=repository,
        max_answer_tokens=settings.llm.generation_max_tokens,
    )
    health_probes: HealthProbes = {
        "sqlite": lambda: repository.active_document_ids(),
        "qdrant": lambda: vector_store.count(),
        "ollama": lambda: _probe_ollama(settings.llm.endpoint),
    }
    return create_app(chat_service, health_probes, settings.api.max_question_chars)


def _probe_ollama(endpoint: str) -> None:
    """생성 없이 서버 생존만 본다. 커넥션은 매번 새로 연다 — 확인용 1회 호출이라
    재사용 규칙(TIME_WAIT)의 대상이 아니고, LLM 클라이언트의 단일 커넥션을
    건드리면 진행 중인 생성과 섞인다."""
    parts = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=3)
    try:
        connection.request("GET", "/api/version")
        response = connection.getresponse()
        if response.status != 200:
            raise ConnectionError(f"HTTP {response.status}")
        response.read()
    finally:
        connection.close()
