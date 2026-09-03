"""화면 런처 인자 조립 테스트 (#29).

이 PR이 내건 보안 주장("런처가 127.0.0.1을 강제한다")을 지키는 자리다.
3차 리뷰에서 실제로 뚫렸다 — 사용자 인자를 고정 플래그 뒤에 붙였더니
`--server.address=0.0.0.0` 하나로 무력화됐다.
"""

from __future__ import annotations

from techdoc_rag.ui.launcher import ADDRESS, build_command, build_environment


def test_고정_주소가_항상_마지막에_온다() -> None:
    """Streamlit은 뒤에 온 --server.address가 이긴다."""
    command = build_command("python", [])

    assert command[-1] == f"--server.address={ADDRESS}"


def test_사용자가_0_0_0_0을_붙여도_고정_주소가_이긴다() -> None:
    command = build_command("python", ["--server.address=0.0.0.0", "--server.port=8600"])

    addresses = [argument for argument in command if argument.startswith("--server.address")]
    assert addresses[-1] == f"--server.address={ADDRESS}"
    assert "--server.port=8600" in command  # 다른 인자는 그대로 살아 있어야 한다


def test_환경변수도_함께_건다() -> None:
    """CLI가 이기므로 이것만으로 강제되지는 않는다. 두 번째 방어선."""
    environment = build_environment({"PATH": "x"})

    assert environment["STREAMLIT_SERVER_ADDRESS"] == ADDRESS
    assert environment["PATH"] == "x"  # 기존 환경을 지우지 않는다
