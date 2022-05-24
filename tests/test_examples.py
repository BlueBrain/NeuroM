import subprocess
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

EXAMPLES_DIR = TESTS_DIR.parent / "examples"
print(EXAMPLES_DIR)

@pytest.mark.parametrize("filename", EXAMPLES_DIR.glob("*.py"))
def test_example(filename):
    result = subprocess.run(["python", filename], capture_output=False)
    assert result.returncode == 0
