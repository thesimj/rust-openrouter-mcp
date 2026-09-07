#!/usr/bin/env python3
"""Run permanent audit regressions without network calls or temporary source edits.

Passing now confirms corrected behavior. The original defect evidence appears
in codebase-audit-2026-09-07.md. Use cargo test --locked for the full HTTP suite.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    sys.exit(subprocess.call(
        ["cargo", "test", "--locked", "--offline", "audit_regression"], cwd=ROOT
    ))
