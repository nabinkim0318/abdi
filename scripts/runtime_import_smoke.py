#!/usr/bin/env python3
"""Prove a clean runtime install can import the live application modules.

Do not import app.py here: it calls Streamlit page config at module import
time. Import the package graph that app.py uses instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import bias_audit_tool  # noqa: F401
    import bias_audit_tool.preprocessing.modeling_pipeline  # noqa: F401
    import bias_audit_tool.utils.ui_helpers  # noqa: F401
    import bias_audit_tool.visualization.ui_blocks  # noqa: F401
    import bias_audit_tool.visualization.visualization  # noqa: F401

    print("runtime import smoke: ok")


if __name__ == "__main__":
    main()
