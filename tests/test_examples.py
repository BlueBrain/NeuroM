import os
import tempfile
import importlib.util
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

EXAMPLES_DIR = TESTS_DIR.parent / "examples"
print(EXAMPLES_DIR)


@pytest.mark.parametrize("filepath", EXAMPLES_DIR.glob("*.py"))
def test_example(filepath):

    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    module = spec.loader.load_module()

    with tempfile.TemporaryDirectory() as tempdir:

        # change directory to avoid creating files in the root folder
        try:
            cwd = os.getcwd()
            os.chdir(tempdir)
            module.main()
        finally:
            os.chdir(cwd)
