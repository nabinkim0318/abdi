"""Contract tests for requirements drift detection and startup smokes."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "scripts" / "requirements-contract.sh"
STREAMLIT_SMOKE = ROOT / "scripts" / "streamlit_startup_smoke.py"


def _run(args, env=None, timeout=20):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_check_requirements_fails_on_stale_export():
    result = _run(
        ["bash", str(CONTRACT), "check"],
        env={
            "ABDI_EXPORT_CMD": 'printf "stale-package==0.0.1\\n" > "$ABDI_EXPORT_OUT"'
        },
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "out of sync" in combined
    assert "make requirements" in combined


def test_check_requirements_fails_when_export_fails():
    result = _run(
        ["bash", str(CONTRACT), "check"],
        env={"ABDI_EXPORT_CMD": "exit 1"},
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "matches the Poetry runtime export" not in combined


def test_check_requirements_fails_when_export_is_empty():
    result = _run(
        ["bash", str(CONTRACT), "check"],
        env={"ABDI_EXPORT_CMD": ': > "$ABDI_EXPORT_OUT"'},
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "empty file" in combined


def test_check_requirements_passes_when_export_matches_committed():
    result = _run(
        ["bash", str(CONTRACT), "check"],
        env={"ABDI_EXPORT_CMD": 'cp requirements.txt "$ABDI_EXPORT_OUT"'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches the Poetry runtime export" in result.stdout + result.stderr


def test_streamlit_startup_smoke_times_out_instead_of_hanging():
    sleeper = f"{sys.executable} -c 'import time; time.sleep(30)'"
    result = _run(
        [sys.executable, str(STREAMLIT_SMOKE)],
        env={
            "ABDI_STREAMLIT_CMD": sleeper,
            "ABDI_STREAMLIT_SMOKE_TIMEOUT": "2",
            "ABDI_STREAMLIT_SMOKE_PORT": "9",
            "ABDI_STREAMLIT_SMOKE_HEALTH_URL": "http://127.0.0.1:9/_stcore/health",
        },
        timeout=15,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "timed out" in combined


def test_streamlit_startup_smoke_fails_if_process_exits_early():
    result = _run(
        [sys.executable, str(STREAMLIT_SMOKE)],
        env={
            "ABDI_STREAMLIT_CMD": f"{sys.executable} -c 'raise SystemExit(3)'",
            "ABDI_STREAMLIT_SMOKE_TIMEOUT": "10",
            "ABDI_STREAMLIT_SMOKE_PORT": "9",
            "ABDI_STREAMLIT_SMOKE_HEALTH_URL": "http://127.0.0.1:9/_stcore/health",
        },
        timeout=15,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "exited early" in combined
