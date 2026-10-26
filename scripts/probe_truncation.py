"""프롬프트가 num_ctx를 넘으면 Ollama가 앞과 뒤 중 어느 쪽을 버리는지 잰다 (#57).

프롬프트 맨 앞과 맨 뒤에 서로 다른 확인 코드를 넣고 사이를 번호 붙은 채움 줄로
채운다. 한도 안(대조군)과 한도 밖으로 같은 프롬프트를 보내서
1. prompt_eval_count가 얼마로 나오는지
2. 모델이 앞 코드와 뒤 코드 중 무엇을 아는지
를 본다. 질문은 맨 뒤에 있으므로 뒤가 잘리면 모델은 질문 자체를 못 본다.

운영 프롬프트는 맨 앞에 "근거에 없는 내용은 답하지 마라" 지시문이 있다. 앞이
잘리는 쪽이면 한도를 넘긴 문항에서 그 지시가 사라진다.

사용:
    python scripts/probe_truncation.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from techdoc_rag.config import load_settings  # noqa: E402

HEAD_CODE = "앞코드-7391"
TAIL_CODE = "뒤코드-2846"
FILLER_LINES = 400


def build_prompt() -> str:
    lines = [f"[맨 앞] 이 문서의 첫 확인 코드는 {HEAD_CODE} 입니다."]
    lines += [
        f"채움 {index:04d}: 인버터 매뉴얼의 일반 안내 문장입니다. 내용은 중요하지 않습니다."
        for index in range(1, FILLER_LINES + 1)
    ]
    lines.append(f"[맨 뒤] 이 문서의 마지막 확인 코드는 {TAIL_CODE} 입니다.")
    lines.append(
        "질문: 위 문서에 나온 확인 코드를 모두 그대로 적으세요. "
        "보이지 않는 코드는 '없음'이라고 적으세요. 그리고 보이는 채움 줄 번호 중 "
        "가장 작은 번호를 적으세요."
    )
    return "\n".join(lines)


def generate(endpoint: str, model: str, prompt: str, num_ctx: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": 120},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    settings = load_settings()
    prompt = build_prompt()
    for num_ctx in (16384, 4096, 2048):
        result = generate(settings.llm.endpoint, settings.llm.model_name, prompt, num_ctx)
        text = result["response"].strip().replace("\n", " ")
        # 모델이 "뒤코드 -2846"처럼 띄어 쓰기도 해서 공백을 빼고 비교한다.
        compact = text.replace(" ", "")
        print(
            f"num_ctx={num_ctx:>5}  prompt_eval_count={result['prompt_eval_count']:>5}  "
            f"앞코드={'O' if HEAD_CODE in compact else 'X'}  "
            f"뒤코드={'O' if TAIL_CODE in compact else 'X'}"
        )
        print(f"    답: {text[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
