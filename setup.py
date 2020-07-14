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
from setuptools import setup
from setuptools import find_packages


setup(
    description='NeuroM: a light-weight neuron morphology analysis package',
    author='Blue Brain Project, EPFL',
    url='http://https://github.com/BlueBrain/NeuroM',
    install_requires=[
        'click>=7.0',
        'h5py>=2.7.1',
        'matplotlib>=3.2.1',
        'numpy>=1.8.0',
        'pandas>=1.0.5',
        'pyyaml>=3.10',
        'scipy>=1.2.0',
        'tqdm>=4.8.4',
        ],
    packages=find_packages(),
    license='BSD',
    scripts=['apps/raw_data_check',
             'apps/morph_check',
             'apps/morph_stats',
             ],
    entry_points={
        'console_scripts': ['neurom=neurom.apps.cli:cli']
    },
    name='neurom',
    extras_require={
        'plotly': ['plotly>=3.6.0'],
    },
    include_package_data=True,
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 6 - Mature',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=['setuptools_scm'],
)
