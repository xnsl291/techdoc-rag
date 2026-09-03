"""화면 실행 진입점 (#29). 주소를 코드에 박아 실행 위치와 무관하게 127.0.0.1에만 연다.

문서로만 "--server.address를 붙여라"라고 하면 조용히 뚫린다. 명령 일부만
복사하거나, 셸 히스토리의 옛 명령을 다시 쓰거나, 래퍼(작업 러너·서비스 등록)가
플래그를 빠뜨리면 Streamlit은 기본값 0.0.0.0으로 열면서 오류를 내지 않는다.
인증이 없는 화면이라 그 실패가 조용하다는 점이 위험하다(리뷰 #30).

환경변수와 플래그를 둘 다 건다. 환경변수는 하위 프로세스가 무엇을 하든 남고,
플래그는 명시적이라 읽는 사람이 의도를 안다.

사용:
    python scripts/run_ui.py            # http://127.0.0.1:8501
    python scripts/run_ui.py --server.port=8600
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ADDRESS = "127.0.0.1"
APP_PATH = Path(__file__).resolve().parents[1] / "src" / "techdoc_rag" / "ui" / "streamlit_app.py"


def main() -> int:
    environment = {**os.environ, "STREAMLIT_SERVER_ADDRESS": ADDRESS}
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        f"--server.address={ADDRESS}",
        *sys.argv[1:],
    ]
    print(f"화면 실행: {ADDRESS} 고정 (API 주소는 TECHDOC_API_URL로 바꿀 수 있음)")
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    sys.exit(main())
