"""Q&A 글의 질문 본문을 받아 온다.

받은 것은 스크래치패드에만 둔다. 저장소에는 원문을 넣지 않는다.
작성자 이메일과 닉네임은 개인정보라 아예 꺼내지 않는다.
"""

import html
import json
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

IDS = [
    30811, 21105, 15533, 21057, 25405, 23814,
    15002, 18351, 17404, 10442, 23818, 16319,
    25065, 26272, 11858, 25229,
]
BASE = "https://ssq.ls-electric.com/api/guest/ssqData/document/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"


def strip_html(raw: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</li>|</div>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def main() -> None:
    for doc_id in IDS:
        req = urllib.request.Request(
            f"{BASE}{doc_id}",
            headers={
                "User-Agent": UA,
                "Referer": f"https://ssq.ls-electric.com/kr/ko/community/qna/document/{doc_id}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
        except Exception as error:  # noqa: BLE001
            print(f"\n===== {doc_id} 실패: {error}")
            continue
        # 제목과 본문만 쓴다. 작성자 관련 칸은 건드리지 않는다.
        print(f"\n===== {doc_id} | {data.get('docType')} | {data.get('docTitle')}")
        print(strip_html(data.get("description") or "")[:1400])
        time.sleep(0.7)  # 남의 서버다. 천천히 부른다


if __name__ == "__main__":
    main()
