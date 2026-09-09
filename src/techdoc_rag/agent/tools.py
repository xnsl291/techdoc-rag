"""에이전트가 부를 수 있는 도구 (#42).

지금까지는 질문 하나에 검색 한 번, 답 하나였다. 그 구조로는 "G100과 M100의
정격 전류를 비교해줘"를 처리할 수 없다. **몇 번을 어떤 순서로 찾을지가 질문마다
다르기** 때문이다. 그래서 찾는 방법을 도구로 내놓고 순서는 모델이 정하게 한다.

도구를 셋으로 나눈 이유.

- `list_documents` — 모델은 어떤 문서가 색인돼 있는지 모른다. 사용자가 "G100"이라
  불러도 그것이 있는지 확인할 방법이 필요하다
- `search_manual` — 한 문서 안에서, 또는 전체에서 근거를 찾는다
- `compare_spec` — **문서마다 따로 검색해** 양쪽 근거를 함께 돌려준다.
  전체 검색 한 번으로는 안 된다. 2026-09-08 실측에서 "정격 전류" 상위 6개가
  전부 G100이었고 M100은 하나도 없었다. 점수가 높은 문서가 상위를 차지한다

**판정은 도구가 하지 않는다.** `compare_spec`은 양쪽 근거를 모아 줄 뿐이고,
같은지 다른지 조건이 다른지는 모델이 근거를 보고 말한다. 값이 다르다는 이유만으로
충돌로 단정하지 않는다는 규칙(DP-39)이 도구 안에 숨으면 검증할 수 없다.

**도구 실패와 인프라 장애를 구분한다.** "그런 문서가 없다"는 도구가 알려 줄 정상
결과이므로 문자열로 돌려준다. 검색 저장소에 못 붙는 것은 장애이므로 예외를
그대로 올린다(D-005). 장애를 도구 결과로 감추면 모델이 "근거가 없다"고 답한다.
"""

from __future__ import annotations

import json

from techdoc_rag.domain.ports import DocumentRepository
from techdoc_rag.query.retriever import Retriever

# 도구가 돌려주는 근거 하나의 길이 상한. 도구 결과는 대화에 계속 쌓이므로
# 청크 전체를 넣으면 몇 번만 불러도 컨텍스트가 찬다.
EVIDENCE_CHARS = 500

# 비교할 때 문서당 가져올 근거 수. 늘리면 문맥이 넓어지지만 문서 수만큼 곱해진다.
EVIDENCE_PER_DOCUMENT = 3


class ToolBox:
    """도구 정의와 실행을 한곳에 둔다.

    정의(스키마)와 구현이 떨어져 있으면 인자 이름이 어긋나도 실행 시점까지
    모른다. 모델은 스키마만 보고 인자를 만들기 때문이다.
    """

    def __init__(self, retriever: Retriever, repository: DocumentRepository) -> None:
        self._retriever = retriever
        self._repository = repository

    def definitions(self) -> list[dict]:
        """Ollama tool calling 스키마."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": (
                        "지금 검색할 수 있는 매뉴얼 목록을 돌려준다. "
                        "사용자가 말한 제품이 있는지 확인할 때 먼저 부른다."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_manual",
                    "description": (
                        "매뉴얼에서 질문과 관련된 근거를 찾는다. "
                        "document를 지정하면 그 문서 안에서만 찾는다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "찾을 내용. 매뉴얼에 쓰일 법한 용어로",
                            },
                            "document": {
                                "type": "string",
                                "description": (
                                    "문서 이름이나 제품명(예: G100). "
                                    "비우면 전체에서 찾는다"
                                ),
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_spec",
                    "description": (
                        "여러 문서에서 같은 항목의 근거를 각각 찾아 함께 돌려준다. "
                        "제품 간 사양을 비교할 때 쓴다. 전체 검색은 한 문서에 "
                        "치우치므로 비교에는 이 도구를 쓴다. 같은지 다른지는 "
                        "돌려받은 근거를 보고 직접 판단한다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "비교할 항목(예: 정격 전류, 주위 온도)",
                            },
                            "documents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "비교할 제품명 목록. 둘 이상",
                            },
                        },
                        "required": ["field", "documents"],
                    },
                },
            },
        ]

    def call(self, name: str, arguments: dict) -> str:
        """도구를 실행하고 모델에 넣을 문자열을 돌려준다."""
        if name == "list_documents":
            return self._list_documents()
        if name == "search_manual":
            return self._search(
                question=arguments.get("question", ""), document=arguments.get("document")
            )
        if name == "compare_spec":
            return self._compare(
                field=arguments.get("field", ""), documents=arguments.get("documents") or []
            )
        # 모델이 없는 도구를 부르는 일이 있다. 예외로 끊지 않고 알려 주면
        # 다음 차례에 스스로 고쳐 부른다.
        return json.dumps({"오류": f"그런 도구가 없음: {name}"}, ensure_ascii=False)

    def _documents(self) -> dict[str, str]:
        """document_id -> 사람이 읽는 이름."""
        names = {}
        for document_id in self._repository.active_document_ids():
            document = self._repository.get(document_id)
            if document is not None:
                names[document_id] = document.original_filename
        return names

    def _resolve(self, wanted: str) -> list[str]:
        """사용자가 부른 이름을 document_id로 바꾼다.

        모델은 "G100"이라 부르고 저장소는 "ls-g100-v1"로 갖고 있다. 대소문자와
        구분자를 지우고 겹치는지 본다. 못 찾으면 빈 목록이고, 그 사실을 도구가
        결과로 알려 준다.
        """
        key = wanted.lower().replace("-", "").replace("_", "").replace(" ", "")
        if not key:
            return []
        matched = []
        for document_id, filename in self._documents().items():
            haystack = (document_id + filename).lower().replace("-", "").replace("_", "")
            if key in haystack.replace(" ", ""):
                matched.append(document_id)
        return matched

    def _list_documents(self) -> str:
        documents = [
            {"id": document_id, "이름": filename}
            for document_id, filename in self._documents().items()
        ]
        return json.dumps({"문서": documents}, ensure_ascii=False)

    def _search(self, question: str, document: str | None) -> str:
        if not question.strip():
            return json.dumps({"오류": "찾을 내용이 비었음"}, ensure_ascii=False)
        document_ids = None
        if document:
            document_ids = self._resolve(document)
            if not document_ids:
                return json.dumps(
                    {"오류": f"그런 문서가 없음: {document}", "힌트": "list_documents로 확인할 것"},
                    ensure_ascii=False,
                )
        result = self._retriever.retrieve(question, document_ids=document_ids)
        return json.dumps(
            {"질의": question, "근거": self._to_evidence(result.chunks)}, ensure_ascii=False
        )

    def _compare(self, field: str, documents: list[str]) -> str:
        if not field.strip():
            return json.dumps({"오류": "비교할 항목이 비었음"}, ensure_ascii=False)
        if len(documents) < 2:
            return json.dumps({"오류": "비교하려면 문서가 둘 이상 필요함"}, ensure_ascii=False)

        found: dict[str, list[dict]] = {}
        missing: list[str] = []
        for wanted in documents:
            document_ids = self._resolve(wanted)
            if not document_ids:
                missing.append(wanted)
                continue
            result = self._retriever.retrieve(field, document_ids=document_ids)
            found[wanted] = self._to_evidence(result.chunks[:EVIDENCE_PER_DOCUMENT])

        payload: dict = {"항목": field, "문서별 근거": found}
        if missing:
            payload["못 찾은 문서"] = missing
        # 판정을 넣지 않는다. 근거만 주고 같은지 다른지는 모델이 말한다(DP-39).
        payload["안내"] = "값이 다르다고 바로 충돌로 보지 말 것. 조건과 단위를 먼저 확인할 것"
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _to_evidence(chunks) -> list[dict]:
        return [
            {
                "문서": item.chunk.document_id,
                "쪽": (
                    f"{item.chunk.page_start}"
                    if item.chunk.page_start == item.chunk.page_end
                    else f"{item.chunk.page_start}~{item.chunk.page_end}"
                ),
                "본문": item.chunk.text[:EVIDENCE_CHARS],
            }
            for item in chunks
        ]
