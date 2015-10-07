# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Distribution configuration for neurom
"""
# pylint: disable=R0801
import os
from setuptools import setup
from setuptools import find_packages
import pip
from pip.req import parse_requirements
from optparse import Option
from neurom.version import VERSION


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    # Hack for old pip versions
    # Versions greater than 1.x have a required parameter "sessions" in
    # parse_requierements
    if pip.__version__.startswith('1.'):
        install_reqs = parse_requirements(reqs_file, options=options)
    else:
        from pip.download import PipSession  # pylint:disable=E0611
        options.isolated_mode = False
        install_reqs = parse_requirements(reqs_file,  # pylint:disable=E1123
                                          options=options,
                                          session=PipSession)

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
    'author': 'BBP Algorithm Development Team',
    'url': 'http://https://github.com/BlueBrain/NeuroM',
    'author_email': 'juan.palacios@epfl.ch, lida.kanari@epfl.ch',
    'version': VERSION,
    'install_requires': REQS,
    'extras_require': EXTRA_REQS,
    'packages': find_packages(),
    'license': 'BSD',
    'scripts': ['apps/raw_data_check',
                'apps/morph_check',
                'apps/morph_stats',
                ],
    'name': 'neurom',
    'include_package_data': True,
}

setup(**config)
