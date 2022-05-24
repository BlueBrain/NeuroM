import os
import tempfile
import subprocess
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

EXAMPLES_DIR = TESTS_DIR.parent / "examples"
print(EXAMPLES_DIR)

@pytest.mark.parametrize("filename", EXAMPLES_DIR.glob("*.py"))
def test_example(filename):

    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tempdir:

        # change directory to avoid creating files in the root folder
        os.chdir(tempdir)

        result = subprocess.run(["python", filename], capture_output=False)
        assert result.returncode == 0

    os.chdir(cwd)
