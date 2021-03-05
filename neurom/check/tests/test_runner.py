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

import os
from collections import OrderedDict
from copy import copy
from pathlib import Path

from neurom.check.runner import CheckRunner
from neurom.exceptions import ConfigError
from nose import tools as nt

_path = os.path.dirname(os.path.abspath(__file__))
SWC_PATH = Path(__file__).parent / '../../../test_data/swc/'


CONFIG = {
    'checks': {
        'structural_checks': [
            'has_soma_points',
            'has_valid_soma',
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

checks_not_in_morphio = OrderedDict([
    ('has_valid_neurites', 'Has valid neurites'),
    ('has_sequential_ids', 'Has sequential ids'),
    ('has_increasing_ids', 'Has increasing ids'),
    ('is_single_tree', 'Is single tree'),
])
CONFIG['checks']['structural_checks'] += list(checks_not_in_morphio.keys())

CONFIG_COLOR = copy(CONFIG)
CONFIG_COLOR['color'] = True


def _run_test(path, ref, config=CONFIG, should_pass=False):
    """Run checkers with the passed "config" on file "path"
    and compare the results to 'ref'"""
    results = CheckRunner(config).run(path)
    nt.assert_dict_equal(dict(results['files'][path]), ref)
    nt.assert_equal(results['STATUS'],
                    "PASS" if should_pass else "FAIL")

ref = dict([
    ("Has soma points", True),
    ("Has valid soma", True),
    ("Has basal dendrite", True),
    ("Has axon", True),
    ("Has apical dendrite", True),
    ("Has all nonzero segment lengths", True),
    ("Has all nonzero section lengths", True),
    ("Has all nonzero neurite radii", True),
    ("Has nonzero soma radius", True),
    ("ALL", True)
])
ref.update({(item.replace('_', ' ').capitalize(), True) for item in checks_not_in_morphio.values()})

def test_ok_neuron():
    _run_test(os.path.join(SWC_PATH, 'Neuron.swc'),
              ref,
              should_pass=True)

def test_ok_neuron_color():
    _run_test(os.path.join(SWC_PATH, 'Neuron.swc'),
              ref,
              CONFIG_COLOR,
              should_pass=True)


def test_zero_length_sections_neuron():
    expected = dict([
                  ("Has soma points", True),
                  ("Has valid soma", True),
                  ("Has basal dendrite", True),
                  ("Has axon", True),
                  ("Has apical dendrite", True),
                  ("Has all nonzero segment lengths", False),
                  ("Has all nonzero section lengths", False),
                  ("Has all nonzero neurite radii", True),
                  ("Has nonzero soma radius", True),
                  ("ALL", False)
    ])
    expected.update({(item, True) for item in checks_not_in_morphio.values()})
    _run_test(os.path.join(SWC_PATH, 'Neuron_zero_length_sections.swc'),
              expected)


def test_single_apical_neuron():
    expected = dict([
                  ("Has soma points", True),
                  ("Has valid soma", True),
                  ("Has basal dendrite", False),
                  ("Has axon", False),
                  ("Has apical dendrite", True),
                  ("Has all nonzero segment lengths", False),
                  ("Has all nonzero section lengths", True),
                  ("Has all nonzero neurite radii", True),
                  ("Has nonzero soma radius", True),
                  ("ALL", False)
              ])
    expected.update({(item, True) for item in checks_not_in_morphio.values()})
    _run_test(os.path.join(SWC_PATH, 'Single_apical.swc'),
              expected)


def test_single_basal_neuron():
    expected = dict(
                  ([
                      ("Has soma points", True),
                      ("Has valid soma", True),
                      ("Has basal dendrite", True),
                      ("Has axon", False),
                      ("Has apical dendrite", False),
                      ("Has all nonzero segment lengths", False),
                      ("Has all nonzero section lengths", True),
                      ("Has all nonzero neurite radii", True),
                      ("Has nonzero soma radius", False),  # Should be True, will be fixed by v2
                      ("ALL", False)
                  ]))
    expected.update({(item, True) for item in checks_not_in_morphio.values()})
    _run_test(os.path.join(SWC_PATH, 'Single_basal.swc'),
              expected)


def test_single_axon_neuron():
    expected = dict([
                  ("Has soma points", True),
                  ("Has valid soma", True),
                  ("Has basal dendrite", False),
                  ("Has axon", True),
                  ("Has apical dendrite", False),
                  ("Has all nonzero segment lengths", False),
                  ("Has all nonzero section lengths", True),
                  ("Has all nonzero neurite radii", True),
                  ("Has nonzero soma radius", True),
                  ("ALL", False)
              ])
    expected.update({(item, True) for item in checks_not_in_morphio.values()})
    _run_test(os.path.join(SWC_PATH, 'Single_axon.swc'),
              expected)


def test_single_apical_no_soma():
    expected = dict([('Has soma points', False),
                    ('Has valid soma', False),
                    ('ALL', False)])
    expected.update({(item, True) for item in checks_not_in_morphio.values()})
    expected['Has valid neurites'] = False
    _run_test(os.path.join(SWC_PATH, 'Single_apical_no_soma.swc'),
              expected)


def test_directory_input():
    checker = CheckRunner(CONFIG)
    summ = checker.run(SWC_PATH)
    nt.eq_(summ['files'][os.path.join(SWC_PATH, 'Single_axon.swc')]['Has axon'], True)
    nt.eq_(summ['files'][os.path.join(SWC_PATH, 'Single_apical.swc')]['Has axon'], False)


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
