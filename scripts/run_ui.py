"""화면 실행 진입점 (#29). 주소를 코드에 박아 실행 위치·인자 실수와 무관하게 127.0.0.1에만 연다.

문서로만 "--server.address를 붙여라"라고 하면 조용히 뚫린다. 명령 일부만
복사하거나, 셸 히스토리의 옛 명령을 다시 쓰거나, 래퍼가 플래그를 빠뜨리면
Streamlit은 기본값 0.0.0.0으로 열면서 오류를 내지 않는다. 인증이 없는
화면이라 그 실패가 조용하다는 점이 위험하다(리뷰 #30).

조립과 우선순위 규칙은 techdoc_rag.ui.launcher에 있고 테스트가 지킨다.

사용:
    python scripts/run_ui.py                    # http://127.0.0.1:8501
    python scripts/run_ui.py --server.port=8600
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techdoc_rag.ui.launcher import (  # noqa: E402
    ADDRESS,
    build_command,
    build_environment,
)


def main() -> int:
    print(f"화면 실행: {ADDRESS} 고정 (API 주소는 TECHDOC_API_URL로 바꿀 수 있음)")
    return subprocess.call(
        build_command(sys.executable, sys.argv[1:]), env=build_environment(os.environ)
    )


if __name__ == "__main__":
    sys.exit(main())
