import textwrap
import warnings
from io import StringIO
from pathlib import Path

import neurom.io as io
import neurom.io.neurolucida as nasc
import numpy as np
from mock import patch
from neurom import load_neuron
from neurom.core.dataformat import COLS
from neurom.io.datawrapper import DataWrapper
from nose.tools import eq_, ok_
from numpy.testing import assert_array_equal

DATA_PATH = Path(Path(__file__).parent, '../../../test_data')
NEUROLUCIDA_PATH = Path(DATA_PATH, 'neurolucida')


def test__match_section():
    # no match in first 5
    section = [0, 1, 2, 3, 4, 'something']
    match = {'Foo': 'Bar', }
    eq_(nasc._match_section(section, match), None)


def test__get_tokens():
    morph_fd = StringIO(u'((()))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', '(', '(', ')', ')', ')'])

    morph_fd = StringIO(u'(Baz("Bar"("Foo")))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', 'Baz', '(', '"Bar"', '(', '"Foo"', ')', ')', ')'])

    morph_fd = StringIO(u'(Baz("Cell Bar Body"("Foo")))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', 'Baz', '(', '"Cell Bar Body"', '(', '"Foo"', ')', ')', ')'])


def test__parse_section():
    with patch('neurom.io.neurolucida._match_section') as mock_match:
        mock_match.return_value = False  # want all sections

        token_iter = iter(['(', '(', '(', ')', ')', ')'])
        section = nasc._parse_section(token_iter)
        eq_(section, [[[[]]]])

        token_iter = iter(['(', 'Baz', '(', '"Bar"', '(', '"Foo"', ')', ')', ')'])
        section = nasc._parse_section(token_iter)
        eq_(section, [['Baz',
                       ['"Bar"',
                        ['"Foo"',
                         ]]]])


def test__parse_sections():
    string_section = textwrap.dedent(
        u"""(FilledCircle
           (Color RGB (64, 0, 128))
           (Name "Marker 11")
           (Set "axons")
           ( -189.59    55.67    28.68     0.12)  ; 1
           )  ;  End of markers

           ( (Color Yellow)
           (Axon)
           (Set "axons")
           (  -40.54  -113.20   -36.61     0.12)  ; Root
           (  -40.54  -113.20   -36.61     0.12)  ; 1, R
           Generated
           )  ;  End of tree
        """)
    morph_fd = StringIO(string_section)
    sections = nasc._parse_sections(morph_fd)
    eq_(len(sections), 1)  # FilledCircle is ignored
    eq_(sections[0], [['Axon'],
                      ['-40.54', '-113.20', '-36.61', '0.12'],
                      ['-40.54', '-113.20', '-36.61', '0.12'],
                      'Generated'])


def test__flatten_section():
    #[X, Y, Z, R, TYPE, ID, PARENT_ID]
    subsection = [['0', '0', '0', '0'],
                  ['1', '1', '1', '1'],
                  ['2', '2', '2', '2'],
                  ['3', '3', '3', '3'],
                  ['4', '4', '4', '4'],
                  'Generated',
                  ]
    ret = np.array([row for row in nasc._flatten_subsection(subsection, 0, offset=0, parent=-1)])
    # correct parents
    ok_(np.allclose(ret[:, COLS.P], np.arange(-1, 4)))
    ok_(np.allclose(ret[:, COLS.ID], np.arange(0, 5)))

    subsection = [['-1', '-1', '-1', '-1'],
                  [['0', '0', '0', '0'],
                   ['1', '1', '1', '1'],
                   ['2', '2', '2', '2'],
                   ['3', '3', '3', '3'],
                   ['4', '4', '4', '4'],
                   '|',
                   ['1', '2', '3', '4'],
                   ['1', '2', '3', '4'],
                   ['1', '2', '3', '4'],
                   ['1', '2', '3', '4'],
                   ['1', '2', '3', '4'], ]
                  ]
    ret = np.array([row for row in nasc._flatten_subsection(subsection, 0, offset=0, parent=-1)])
    # correct parents
    eq_(ret[0, COLS.P], -1.)
    eq_(ret[1, COLS.P], 0.0)
    eq_(ret[6, COLS.P], 0.0)
    ok_(np.allclose(ret[:, COLS.ID], np.arange(0, 11)))  # correct ID

    # Try a non-standard bifurcation, ie: missing '|' separator
    subsection = [['-1', '-1', '-1', '-1'],
                  [['0', '0', '0', '0'],
                   ['1', '1', '1', '1'], ]
                  ]
    ret = np.array([row for row in nasc._flatten_subsection(subsection, 0, offset=0, parent=-1)])
    eq_(ret.shape, (3, 7))

    # try multifurcation
    subsection = [['-1', '-1', '-1', '-1'],
                  [['0', '0', '0', '0'],
                   ['1', '1', '1', '1'],
                   '|',
                   ['2', '2', '2', '2'],
                   ['3', '3', '3', '3'],
                   '|',
                   ['4', '4', '4', '4'],
                   ['5', '5', '5', '5'], ]
                  ]
    ret = np.array([row for row in nasc._flatten_subsection(subsection, 0, offset=0, parent=-1)])
    # correct parents
    eq_(ret[0, COLS.P], -1.)
    eq_(ret[1, COLS.P], 0.0)
    eq_(ret[3, COLS.P], 0.0)
    eq_(ret[5, COLS.P], 0.0)
    ok_(np.allclose(ret[:, COLS.ID], np.arange(0, 7)))  # correct ID


def test__extract_section():
    section = ['"CellBody"',
               ['CellBody'],
               ['-1', '-1', '-1', '-1'],
               ['1', '1', '1', '1'],
               ]
    section = nasc._extract_section(section)

    # unknown type
    section = ['"Foo"',
               ['Bar'],
               ['-1', '-1', '-1', '-1'],
               ['1', '1', '1', '1'],
               ]
    section = nasc._extract_section(section)


def test_sections_to_raw_data():
    # from my h5 example neuron
    # https://developer.humanbrainproject.eu/docs/projects/morphology-documentation/0.0.2/h5v1.html
    soma = ['"CellBody"',
            ['CellBody'],
            ['1', '1', '0', '.1'],
            ['-1', '1', '0', '.1'],
            ['-1', '-1', '0', '.1'],
            ['1', '-1', '0', '.1'],
            ]
    axon = [['Axon'],
            ['0', '5', '0', '.1'],
            ['2', '9', '0', '.1'],
            ['0', '13', '0', '.1'],
            ['2', '13', '0', '.1'],
            ['4', '13', '0', '.1'],
            ]
    dendrite = [['Dendrite'],
                ['3', '-4', '0', '.1'],
                ['3', '-6', '0', '.1'],
                ['3', '-8', '0', '.1'],
                ['3', '-10', '0', '.1'],
                [['0', '-10', '0', '.1'],
                 '|',
                 ['6', '-10', '0', '.1'],
                 ]
                ]
    fake_neurite = [['This is not ', ], ['a neurite']]
    sections = [soma, fake_neurite, axon, dendrite, ]
    raw_data = nasc._sections_to_raw_data(sections)
    eq_(raw_data.shape, (15, 7))
    ok_(np.allclose(raw_data[:, COLS.ID], np.arange(0, 15)))  # correct ID
    # 3 is ID of end of the soma, 2 sections attach to this
    ok_(np.count_nonzero(raw_data[:, COLS.P] == 3),  2)


# what I think the
# https://developer.humanbrainproject.eu/docs/projects/morphology-documentation/0.0.2/h5v1.html
# would look like
MORPH_ASC = textwrap.dedent(
    u"""\
; Generated by the hand of mgevaert
("CellBody"
  (CellBody)
  (1 1 0 0)  ; 1, 1
  (-1 1 0 0)  ; 1, 2
  (-1 -1 0 0)  ; 1, 3
  (1 -1 0 0)  ; 1, 4
);

((Axon)
 (0 5 0 2)
 (2 9 0 2)
 (0 13 0 2)
 (2 13 0 2)
 (4 13 0 2)
)

((Dendrite)
 (3 -4 0 2)
 (3 -6 0 2)
 (3 -8 0 2)
 (3 -10 0 2)
 (
    (3 -10 0 2)
    (0 -10 0 2)
    (-3 -10 0 2)
 |
    (3 -10 0 2)
    (6 -10 0 2)
    (9 -10 0 2)
 )
)
""")


def test_read():
    with warnings.catch_warnings(record=True):
        rdw = io.load_data(StringIO(MORPH_ASC), reader='asc')
    raw_data = rdw.data_block

    eq_(raw_data.shape, (19, 7))
    ok_(np.allclose(raw_data[:, COLS.ID], np.arange(0, 19)))  # correct ID
    # 3 is ID of end of the soma, 2 sections attach to this
    ok_(np.count_nonzero(raw_data[:, COLS.P] == 3),  2)

    with warnings.catch_warnings(record=True):
        neuron = load_neuron(StringIO(MORPH_ASC), reader='asc')
    assert_array_equal(neuron.neurites[0].root_node.points[:, COLS.XYZ],
                       [[ 0.,  5.,  0.],
                        [ 2.,  9.,  0.],
                        [ 0., 13.,  0.],
                        [ 2., 13.,  0.],
                        [ 4., 13.,  0.]])


def test_load_neurolucida_ascii():
    f = Path(NEUROLUCIDA_PATH, 'sample.asc')
    with warnings.catch_warnings(record=True):
        ascii = io.load_data(f)
    ok_(isinstance(ascii, DataWrapper))
    eq_(len(ascii.data_block), 18)

def test_spine():
    with warnings.catch_warnings(record=True):
        n = load_neuron(Path(NEUROLUCIDA_PATH, 'spine.asc'))

    assert_array_equal(n.neurites[0].points,
                       [[ 0. ,  5. ,  0. ,  1. ],
                        [ 2. ,  9. ,  0. ,  1. ],
                        [ 0. , 13. ,  0. ,  1. ]])

    # with warnings.catch_warnings(record=True):
    #     assert_raises(RawDataError,
    #                   load_neuron, Path(NEUROLUCIDA_PATH, 'broken-spine.asc'))
