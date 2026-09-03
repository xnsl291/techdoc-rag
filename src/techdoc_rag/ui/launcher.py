"""화면 실행 인자 조립 (#29).

`scripts/run_ui.py`가 이것만 부른다. 조립을 분리한 이유는 하나다 —
"런처가 127.0.0.1을 강제한다"가 이 PR이 내건 보안 주장이고, 주장은
테스트로 지켜져야 한다. 서브프로세스를 띄우는 코드 안에 묻어 두면
검증할 수 없다(리뷰 #30 3차에서 실제로 뚫린 뒤 분리함).

Streamlit의 우선순위는 **CLI 플래그 > 환경변수**다. 그래서 고정 주소를
사용자 인자 뒤에 두어 마지막에 오게 한다. 앞에 두면 사용자가
`--server.address=0.0.0.0` 하나만 붙여도 그대로 열린다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

ADDRESS = "127.0.0.1"
APP_PATH = Path(__file__).resolve().parent / "streamlit_app.py"


def build_command(python_executable: str, extra_args: Sequence[str]) -> list[str]:
    """streamlit 실행 명령을 만든다. 고정 주소가 항상 마지막에 온다."""
    return [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        *extra_args,
        f"--server.address={ADDRESS}",
    ]


def build_environment(base: Mapping[str, str]) -> dict[str, str]:
    """환경변수도 함께 건다.

    CLI 플래그가 이기므로 이것만으로는 강제가 안 된다. 플래그가 어떤 경로로든
    빠졌을 때(래퍼가 명령을 다시 조립하는 경우 등)의 두 번째 방어선이다.
    """
    return {**base, "STREAMLIT_SERVER_ADDRESS": ADDRESS}
