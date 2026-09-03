"""화면 실행 진입점 (#29). 주소를 코드에 박아 실행 위치·인자 실수와 무관하게 127.0.0.1에만 연다.

문서로만 "--server.address를 붙여라"라고 하면 조용히 뚫린다. 명령 일부만
복사하거나, 셸 히스토리의 옛 명령을 다시 쓰거나, 래퍼가 플래그를 빠뜨리면
Streamlit은 기본값 0.0.0.0으로 열면서 오류를 내지 않는다. 인증이 없는
화면이라 그 실패가 조용하다는 점이 위험하다(리뷰 #30).

조립과 우선순위 규칙은 techdoc_rag.ui.launcher에 있고 테스트가 지킨다.

패키지가 설치돼 있어야 한다(`pip install -e .`). sys.path를 여기서 손대지 않는
이유는, 그렇게 해도 자식 프로세스인 streamlit에는 전달되지 않아 "런처는 뜨는데
화면이 ModuleNotFoundError로 죽는" 어긋난 상태가 되기 때문이다(리뷰 #30 4차).

사용:
    python scripts/run_ui.py                    # http://127.0.0.1:8501
    python scripts/run_ui.py --server.port=8600
"""

from __future__ import annotations

import os
import subprocess
import sys

from techdoc_rag.ui.launcher import ADDRESS, build_command, build_environment


def main() -> int:
    print(f"화면 실행: {ADDRESS} 고정 (API 주소는 TECHDOC_API_URL로 바꿀 수 있음)")
    return subprocess.call(
        build_command(sys.executable, sys.argv[1:]), env=build_environment(os.environ)
    )


if __name__ == "__main__":
    sys.exit(main())
