#!/usr/bin/env python3
"""Bounded headless Streamlit startup smoke for dependency completeness."""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT = 60
HEALTH_PATH = "/_stcore/health"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read().decode("utf-8", errors="replace").strip().lower()
            return response.status == 200 and (
                body in {"ok", "healthy"} or "ok" in body
            )
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _print_log(path: str) -> None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            print(handle.read(), file=sys.stderr)
    except OSError:
        return


def main() -> int:
    timeout = int(os.environ.get("ABDI_STREAMLIT_SMOKE_TIMEOUT", DEFAULT_TIMEOUT))
    port = int(os.environ.get("ABDI_STREAMLIT_SMOKE_PORT", "0")) or _free_port()
    health_url = os.environ.get(
        "ABDI_STREAMLIT_SMOKE_HEALTH_URL",
        f"http://127.0.0.1:{port}{HEALTH_PATH}",
    )
    override = os.environ.get("ABDI_STREAMLIT_CMD")
    if override:
        cmd = shlex.split(override)
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ]

    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    log_file = tempfile.NamedTemporaryFile(
        prefix="abdi-streamlit-smoke-",
        suffix=".log",
        delete=False,
    )
    log_path = log_file.name
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_file.flush()
                _print_log(log_path)
                print(
                    "streamlit startup smoke: process exited early with "
                    f"code {proc.returncode}",
                    file=sys.stderr,
                )
                return 1
            if _health_ok(health_url):
                print(f"streamlit startup smoke: ok ({health_url})")
                return 0
            time.sleep(0.5)
        log_file.flush()
        _print_log(log_path)
        print(
            "streamlit startup smoke: timed out after "
            f"{timeout}s waiting for {health_url}",
            file=sys.stderr,
        )
        return 1
    finally:
        _terminate(proc)
        log_file.close()
        try:
            os.unlink(log_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
