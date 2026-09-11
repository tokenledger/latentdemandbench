"""The paper's prompt appendix must match the canonical harness contract."""

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generated_prompt_contract_matches_reference_digest(tmp_path):
    expected = '6f0f6e0f7b58a0c7998933aaefbc36722315432586e2988b31215316642b0577'
    out = tmp_path / "prompt_appendix.tex"
    result = subprocess.run(
        [sys.executable, "tools/prompt_appendix.py", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(out.read_bytes()).hexdigest() == expected
