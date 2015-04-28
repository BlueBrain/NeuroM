""" Distribution configuration for neurom
"""
# pylint: disable=R0801
import os
from setuptools import setup
from pip.req import parse_requirements
from optparse import Option
from neurom.version import VERSION


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    install_reqs = parse_requirements(reqs_file, options=options)
    return [str(ir.req) for ir in install_reqs]


BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = parse_reqs(os.path.join(BASEDIR, 'requirements.txt'))

EXTRA_REQS_PREFIX = 'requirements_'
EXTRA_REQS = {}

for file_name in os.listdir(BASEDIR):
    if not file_name.startswith(EXTRA_REQS_PREFIX):
        continue
    base_name = os.path.basename(file_name)
    (extra, _) = os.path.splitext(base_name)
    extra = extra[len(EXTRA_REQS_PREFIX):]
    EXTRA_REQS[extra] = parse_reqs(file_name)

config = {
    'description': 'NeuroM: a light-weight neuron morphology analysis package',
    'author': 'HBP Algorithm Development Team',
    'url': 'http://www.example.com',
    'author_email': 'juan.palacios@epfl.ch, lida.kanari@epfl.ch',
    'version': VERSION,
    'install_requires': REQS,
    'extras_require': EXTRA_REQS,
    'packages': ['neurom', 'neurom.io', 'neurom.core', 'neurom.view'],
    'scripts': [],
    'name': 'neurom',
    'include_package_data': True,
}

setup(**config)
