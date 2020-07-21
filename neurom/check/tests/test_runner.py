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

from pathlib import Path
from copy import copy

from nose import tools as nt

from neurom.check.runner import CheckRunner
from neurom.exceptions import ConfigError
from pathlib import Path


SWC_PATH = Path(__file__).parent.parent.parent.parent / 'test_data/swc/'
NRN_PATH_0 = str(Path(SWC_PATH, 'Neuron.swc'))
NRN_PATH_1 = str(Path(SWC_PATH, 'Neuron_zero_length_sections.swc'))
NRN_PATH_2 = str(Path(SWC_PATH, 'Single_apical.swc'))
NRN_PATH_3 = str(Path(SWC_PATH, 'Single_basal.swc'))
NRN_PATH_4 = str(Path(SWC_PATH, 'Single_axon.swc'))
NRN_PATH_5 = str(Path(SWC_PATH, 'Single_apical_no_soma.swc'))

CONFIG = {
    'checks': {
        'structural_checks': [
            'is_single_tree',
            'has_soma_points',
            'has_sequential_ids',
            'has_increasing_ids',
            'has_valid_soma',
            'has_valid_neurites'
        ],
        'neuron_checks': [
            'has_basal_dendrite',
            'has_axon',
            'has_apical_dendrite',
            'has_all_nonzero_segment_lengths',
            'has_all_nonzero_section_lengths',
            'has_all_nonzero_neurite_radii',
            'has_nonzero_soma_radius'
        ]
    },
    'options': {
        'has_nonzero_soma_radius': 0.0,
        "has_all_nonzero_neurite_radii": 0.007,
        "has_all_nonzero_segment_lengths": 0.01,
        "has_all_nonzero_section_lengths": [0.01]
    },
}

CONFIG_COLOR = copy(CONFIG)
CONFIG_COLOR['color'] = True

REF_0 = {
    'files': {
        NRN_PATH_0: {
            "Is single tree": True,
            "Has soma points": True,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": True,
            "Has valid neurites": True,
            "Has basal dendrite": True,
            "Has axon": True,
            "Has apical dendrite": True,
            "Has all nonzero segment lengths": True,
            "Has all nonzero section lengths": True,
            "Has all nonzero neurite radii": True,
            "Has nonzero soma radius": True,
            "ALL": True
        }
    },
    "STATUS": "PASS"
}

REF_1 = {
    'files': {
        NRN_PATH_1: {
            "Is single tree": True,
            "Has soma points": True,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": True,
            "Has valid neurites": True,
            "Has basal dendrite": True,
            "Has axon": True,
            "Has apical dendrite": True,
            "Has all nonzero segment lengths": False,
            "Has all nonzero section lengths": False,
            "Has all nonzero neurite radii": True,
            "Has nonzero soma radius": True,
            "ALL": False
        }
    },
    "STATUS": "FAIL"
}

REF_2 = {
    'files': {
        NRN_PATH_2: {
            "Is single tree": True,
            "Has soma points": True,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": True,
            "Has valid neurites": True,
            "Has basal dendrite": False,
            "Has axon": False,
            "Has apical dendrite": True,
            "Has all nonzero segment lengths": False,
            "Has all nonzero section lengths": True,
            "Has all nonzero neurite radii": True,
            "Has nonzero soma radius": True,
            "ALL": False
        }
    },
    "STATUS": "FAIL"
}

REF_3 = {
    'files': {
        NRN_PATH_3: {
            "Is single tree": True,
            "Has soma points": True,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": True,
            "Has valid neurites": True,
            "Has basal dendrite": True,
            "Has axon": False,
            "Has apical dendrite": False,
            "Has all nonzero segment lengths": False,
            "Has all nonzero section lengths": True,
            "Has all nonzero neurite radii": True,
            "Has nonzero soma radius": False,
            "ALL": False
        }
    },
    "STATUS": "FAIL"
}

REF_4 = {
    'files': {
        NRN_PATH_4: {
            "Is single tree": True,
            "Has soma points": True,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": True,
            "Has valid neurites": True,
            "Has basal dendrite": False,
            "Has axon": True,
            "Has apical dendrite": False,
            "Has all nonzero segment lengths": False,
            "Has all nonzero section lengths": True,
            "Has all nonzero neurite radii": True,
            "Has nonzero soma radius": True,
            "ALL": False
        }
    },
    "STATUS": "FAIL"
}


REF_5 = {
    'files': {
        NRN_PATH_5: {
            "Is single tree": True,
            "Has soma points": False,
            "Has sequential ids": True,
            "Has increasing ids": True,
            "Has valid soma": False,
            "Has valid neurites": False,
            "ALL": False
        }
    },
    "STATUS": "FAIL"
}


def test_ok_neuron():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_0)
    nt.assert_equal(summ, REF_0)


def test_ok_neuron_color():
    checker = CheckRunner(CONFIG_COLOR)
    summ = checker.run(NRN_PATH_0)
    nt.assert_equal(summ, REF_0)


def test_zero_length_sections_neuron():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_1)
    nt.assert_equal(summ, REF_1)


def test_single_apical_neuron():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_2)
    nt.assert_equal(summ, REF_2)


def test_single_basal_neuron():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_3)
    nt.assert_equal(summ, REF_3)


def test_single_axon_neuron():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_4)
    nt.assert_equal(summ, REF_4)


def test_single_apical_no_soma():
    checker = CheckRunner(CONFIG)
    summ = checker.run(NRN_PATH_5)
    nt.assert_equal(summ, REF_5)


def test_directory_input():
    checker = CheckRunner(CONFIG)
    summ = checker.run(SWC_PATH)
    nt.eq_(summ['files'][NRN_PATH_0]['Has axon'], True)
    nt.eq_(summ['files'][NRN_PATH_2]['Has axon'], False)


@nt.raises(IOError)
def test_invalid_data_path_raises_IOError():
    checker = CheckRunner(CONFIG)
    _ = checker.run('foo/bar/baz')


def test__sanitize_config():
    # fails if missing 'checks'
    nt.assert_raises(ConfigError, CheckRunner._sanitize_config, {})

    # creates minimal config
    new_config = CheckRunner._sanitize_config({'checks': {}})
    nt.eq_(new_config, {'checks':
                        {'structural_checks': [],
                         'neuron_checks': [],
                         },
                        'options': {},
                        'color': False,
                        })

    # makes no changes to already filled out config
    new_config = CheckRunner._sanitize_config(CONFIG)
    nt.eq_(CONFIG, new_config)
