import json
from pathlib import Path
import pkg_resources
from neurom.apps import get_config
from neurom.check.runner import CheckRunner

EXAMPLE_CONFIG = Path(pkg_resources.resource_filename('neurom', 'config'), 'morph_check.yaml')


def main(datapath, config, output):
    config = get_config(config, EXAMPLE_CONFIG)
    checker = CheckRunner(config)
    summary = checker.run(datapath)
    with open(output, 'w') as json_output:
        json.dump(summary, json_output, indent=4)
