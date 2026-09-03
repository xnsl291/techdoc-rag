# techdoc-rag

산업 기술문서를 하나의 지식기반으로 관리하고, **근거 기반 질의응답**과 **설계변수 구조화 추출**을
제공하는 Local AI 시스템.

원본 문서를 Source of Truth로 두고, 모든 답변에 문서명과 페이지 단위 근거를 붙이는 것을 전제로 설계함.

## 두 개의 Use Case

대상 문서군과 출력 계약이 서로 다르므로 파이프라인을 분리함.

| | UC-1 Manual / Technical QA | UC-2 Specification Extraction |
|---|---|---|
| 대상 문서 | 제품, 기술 매뉴얼 | 시방서, 설계문서 |
| 하는 일 | 자연어 질의 → Retrieval → Grounded Answer + Citation | 주요 설계변수를 정해진 스키마로 추출 |
| 근거가 없을 때 | No-answer 반환 | 필드 단위 `MISSING` |
| Canonical Schema | 필수 아님 | 핵심 |

추출 결과가 안정화된 뒤에는 문서 간 비교로 확장할 수 있음. 값이 다르다는 이유만으로
충돌로 판정하지 않고, 비교 조건(동일 필드 / 비교 가능한 대상 / 버전 관계 / 운전 조건 / 단위 정규화)을
모두 확인한 뒤에만 `CONFLICT_CANDIDATE`로 올림. 최종 판단은 사람이 함.

## 구성

```
Operator ──> Ingestion CLI ──> PDF Parser ──> Chunker ──> Embedding ──> Qdrant
                   └──> SQLite (문서 메타데이터)
                   └──> Local File Storage (원본 PDF)

User ──> Streamlit ──> FastAPI ──> Chat Service ──> Qdrant
                                        └──> Context Builder ──> Ollama ──> Local LLM
                                                                      └──> Answer + Citation
```

운영자가 문서를 사전 등록하고 사용자는 준비된 지식기반에 질의하는 구조.
사용자 업로드는 범위에 없음.

| 계층 | 선택 |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| RAG orchestration | LlamaIndex |
| Vector Store | Qdrant |
| Metadata | SQLite |
| Inference | Ollama + Qwen3.5 (4B / 9B Q4) |

## 설계 원칙

- 원본 문서가 Source of Truth. 벡터와 청크는 언제든 재생성 가능한 파생 데이터로 취급
- 모든 답변은 `document_id / version / page / chunk_id`까지 추적 가능해야 함
- Parsing, Retrieval, Generation 실패를 구분함. 검색이 실패했을 때 LLM의 일반 지식으로 우회하지 않음
- LLM, Embedding, Vector Store는 어댑터로 분리해 상위 기능 수정 없이 교체 가능하게 둠
- 측정하지 않은 수치를 성과로 쓰지 않음. `[실측]`과 `[추정]`을 구분해 표기함

## 현재 상태

**구현 착수 단계.** 설계와 환경 구축이 끝났고 파이프라인 구현은 시작 전임.

| 항목 | 상태 |
|---|---|
| 시스템 설계 | 완료 |
| 개발 환경 | 완료 (Python 3.12, Ollama, Qwen3.5 4B / 9B Q4) |
| LLM 성능 실측 | 부분 완료 |
| Ingestion 파이프라인 | 미착수 |
| Query 파이프라인 | 미착수 |
| 평가셋 | 미작성 |

### 측정 결과

개발 장비: LG gram 16T90SP (Core Ultra 7 155H, RAM 31.7GB, Intel Arc 내장 그래픽)

| 조건 | TTFT 중앙값 | tok/s 중앙값 | 표본 |
|---|---|---|---|
| Qwen3.5 4B Q4, thinking OFF, 512 토큰 | 0.35s | 7.25 | 8회 |

- Ollama는 이 장비에서 **100% CPU로 동작**함. Intel Arc 내장 그래픽 가속은 적용되지 않음
- 9B 및 thinking ON 조건은 측정 진행 중

측정 스크립트는 `scripts/benchmark_llm.py`.

## 실행

문서를 색인한 뒤 API를 띄우고 화면을 연다. Ollama가 먼저 떠 있어야 함.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 1) 매뉴얼 색인 (문서 하나)
.venv\Scripts\python.exe scripts\verify_ingestion_real.py --pdf <매뉴얼.pdf>

# 2) API — 127.0.0.1에만 연다. 인증이 없으므로 외부에 노출하지 않음
.venv\Scripts\python.exe -m uvicorn --factory techdoc_rag.api.app:create_default_app \
    --host 127.0.0.1 --port 8000 --app-dir src

# 3) 화면 (다른 터미널에서)
.venv\Scripts\python.exe scripts\run_ui.py
```

화면은 `scripts/run_ui.py`로 띄운다. 주소가 스크립트에 박혀 있어 실행 위치나 명령 실수와
무관하게 `127.0.0.1`에만 열린다. Streamlit을 직접 실행하려면 `--server.address=127.0.0.1`을
반드시 붙일 것 — `.streamlit/config.toml`에도 같은 값이 있지만 Streamlit이 그 파일을
**실행 디렉터리 기준**으로 찾기 때문에, 저장소 루트가 아닌 곳에서 띄우면 기본값
`0.0.0.0`으로 조용히 되돌아감(인증이 없는 화면이라 노출되면 대가가 큼).

벤치마크 스크립트만 돌리려면:

```bash
.venv\Scripts\python.exe scripts\benchmark_llm.py \
    --models qwen3.5:4b-q4_K_M --repeat 10 --max-tokens 512
```

## 대상 문서

LS ELECTRIC이 다운로드 센터에 공개한 저압 드라이브(인버터) 국문 매뉴얼을 사용함.
평가 질문은 같은 제조사의 공개 Q&A 게시판에 올라온 실제 질문에서 수집함.
질문을 직접 만들면 시스템이 잘 푸는 것만 만들게 되어 평가셋이 편향되기 때문.

수집 스크립트는 `scripts/collect_manuals.py`. 요청 간격을 두고 식별 가능한 User-Agent를 보냄.

### 저작권

**대상 매뉴얼과 Q&A 게시글의 저작권은 각 권리자에게 있음. 이 저장소는 원문을 포함하지 않음.**
파일명, 페이지, 출처 URL만 기록하며 원문은 스크립트로 각자 내려받는 구조임.

이 시스템은 **단일 장비에서 개인이 실험과 평가 목적으로 실행하는 것을 전제**로 함.
수집한 자료를 재배포하거나 이 시스템을 공개 서비스로 배포하지 않음.
