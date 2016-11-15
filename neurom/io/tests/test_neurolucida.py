import tempfile
import textwrap
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
from nose.tools import ok_, eq_, assert_raises
from mock import patch

import neurom.io as io
from neurom.io.datawrapper import DataWrapper, BlockNeuronBuilder
import neurom.io.neurolucida as nasc
from neurom.core.dataformat import COLS

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
NEUROLUCIDA_PATH = os.path.join(DATA_PATH, 'neurolucida')


def test__get_tokens():
    morph_fd = StringIO('((()))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', '(', '(', ')', ')', ')'])

    morph_fd = StringIO('(Baz("Bar"("Foo")))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', 'Baz', '(', '"Bar"', '(', '"Foo"', ')', ')', ')'])

    morph_fd = StringIO('(Baz("Cell Bar Body"("Foo")))')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', 'Baz', '(', '"Cell Bar Body"', '(', '"Foo"', ')', ')', ')'])

    morph_fd = StringIO(
'''
(
    <(spine)>
)
''')
    tokens = list(nasc._get_tokens(morph_fd))
    eq_(tokens, ['(', ')'])


def test__parse_section():
    token_iter = iter(['(', 'Baz', '(', '"Bar"', '(', '"Foo"', ')', ')', ')'])
    section = nasc._parse_section(token_iter)
    eq_(section, [['Baz',
                   ['"Bar"',
                    ['"Foo"',
                     ]]]])


def test__extract_section_points():
    subsection = [['1', '1', '0', '2'],
                  ['-1', '1', '0', '2'],
                  ['1', '-1', '0', '2']]
    points = nasc._extract_section_points(subsection)
    eq_(points, [(1.0, 1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0)])

    subsection = [[[]]]
    points = nasc._extract_section_points(subsection)
    eq_(points, [])

    subsection = ['Low', 'Generated', 'High']
    points = nasc._extract_section_points(subsection)
    eq_(points, [])

    subsection = [['1', '1', '0', '2', 'S1'],
                  ['-1', '1', '0', '2', 'S2'],
                  ['1', '-1', '0', '2', 'S3']]
    points = nasc._extract_section_points(subsection)
    eq_(points, [(1.0, 1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0)])

    nasc._extract_section_points(['Foo'])
    nasc._extract_section_points([['1', '-1', '0', '2', 'NotS']])


def test_find_furcations():
    rows = [['-1', '-1', '-1', '-1'],
            ['0', '0', '0', '0'],
            '|',
            ['1', '2', '3', '4'],
            '|',
            ['1', '2', '3', '4'],
            ]
    furcations = nasc._find_furcations(rows)
    eq_(furcations, [slice(0, 2), slice(3, 4), slice(5, 6), ])


def test_read_subsection():
    neuron_builder = BlockNeuronBuilder()
    id_, parent_id, section_type = 0, -1, 2
    subsection = [['1', '1', '0', '0'],
                  ['-1', '1', '0', '0'],
                  ['1', '-1', '0', '0']]
    new_id = nasc.read_subsection(neuron_builder, id_, parent_id, section_type, subsection)
    eq_(new_id, 1)  # 1 larger than 0
    eq_(neuron_builder.sections[0].parent_id, -1)
    eq_(neuron_builder.sections[0].section_type, 2)
    np.testing.assert_almost_equal(neuron_builder.sections[0].points,
                                   np.array(subsection).astype('float'))

    neuron_builder = BlockNeuronBuilder()
    subsection = [['0', '0', '0', '0'],
                  ['1', '1', '1', '1'],
                  ['2', '2', '2', '2'],
                  ['3', '3', '3', '3'],
                  ['4', '4', '4', '4'],
                  ]
    new_id = nasc.read_subsection(neuron_builder, id_, parent_id, section_type, subsection,
                                  parent_point=[1, 2, 3, 4])
    eq_(new_id, 1)  # 1 larger than 0
    np.testing.assert_almost_equal(neuron_builder.sections[0].points[0],
                                   [1, 2, 3, 4])

    neuron_builder = BlockNeuronBuilder()
    subsection = [['-290.87', '-113.09', '-16.32', '2.06'],  #id = 0, parent = -1
                  ['-290.87', '-113.09', '-16.32', '2.06'],
                  [['-277.14', '-119.13', '-18.02', '0.69'], #id = 1, parent = 0
                   ['-275.54', '-119.99', '-16.67', '0.69'],
                   'Normal',
                   '|',
                   ['-277.80', '-120.28', '-19.48', '0.92'], #id = 2, parent = 0
                   ['-276.65', '-121.14', '-20.20', '0.92'],
                   [['-267.94', '-128.61', '-22.57', '0.69'], #id = 3, parent = 2
                    ['-204.90', '-157.63', '-42.45', '0.69'],
                    'Incomplete',
                    '|',
                    ['-269.77', '-129.47', '-22.57', '0.92'], #id = 4, parent = 2
                    ['-268.17', '-130.62', '-24.75', '0.92'],
                    ['-266.79', '-131.77', '-26.13', '0.92'],
                    'Incomplete']]]
    nasc.read_subsection(neuron_builder, id_, -1, section_type, subsection)
    sections = neuron_builder.sections
    eq_(sections[0].points[0][0], -290.87)
    eq_(sections[1].points[1][0], -277.14)
    eq_(sections[2].points[1][0], -277.80)
    eq_(sections[3].points[1][0], -267.94)
    eq_(sections[4].points[1][0], -269.77)


def test__top_level_sections():
    string_section = textwrap.dedent(
'''(FilledCircle
    (Color RGB (64, 0, 128))
    (Name "Marker 11")
    (Set "axons")
    ( -189.59    55.67    28.68     0.12)  ; 1
   )  ;  End of markers
        ''')
    morph_fd = StringIO(string_section)
    sections = list(nasc._top_level_sections(morph_fd))

    string_section = textwrap.dedent(
'''(FilledCircle
    (Color RGB (64, 0, 128))
    (Name "Marker 11")
    (Set "axons")
    ( -189.59    55.67    28.68     0.12)  ; 1
   )  ;  End of markers
    
   ((Color Yellow)
    (Axon)
    (Set "axons")
    (  -40.54  -113.20   -36.61     0.12)  ; Root
    (  -40.54  -113.20   -36.61     0.12)  ; 1, R
    Generated
   )  ;  End of tree
''')
    morph_fd = StringIO(string_section)
    sections = list(nasc._top_level_sections(morph_fd))
    eq_(len(sections), 1)  # FilledCircle is ignored
    eq_(sections[0], [['Axon'],
                      ['-40.54', '-113.20', '-36.61', '0.12'],
                      ['-40.54', '-113.20', '-36.61', '0.12'],
                      'Generated'])

    string_section = textwrap.dedent('''
( (Color White)  ; [10,1]
  (Dendrite)
  ( -290.87  -113.09   -16.32     2.06)  ; Root
  ( -290.87  -113.09   -16.32     2.06)  ; R, 1
  (
    ( -277.14  -119.13   -18.02     0.69)  ; R-1, 1
    ( -275.54  -119.99   -16.67     0.69)  ; R-1, 2

    (Cross  ;  [3,3]
      (Color Orange)
      (Name "Marker 3")
      ( -271.87  -121.14   -16.27     0.69)  ; 1
      ( -269.34  -122.29   -15.48     0.69)  ; 2
    )  ;  End of markers
     Normal
  |
    ( -277.80  -120.28   -19.48     0.92)  ; R-2, 1
    ( -276.65  -121.14   -20.20     0.92)  ; R-2, 2

    (Cross  ;  [3,3]
      (Color Orange)
      (Name "Marker 3")
      ( -279.41  -119.99   -18.00     0.46)  ; 1
      ( -272.98  -126.60   -21.22     0.92)  ; 2
    )  ;  End of markers
    (
      ( -267.94  -128.61   -22.57     0.69)  ; R-2-1, 1
      ( -204.90  -157.63   -42.45     0.69)  ; R-2-1, 34

      (Cross  ;  [3,3]
        (Color Orange)
        (Name "Marker 3")
        ( -223.67  -157.92   -42.45     0.69)  ; 1
        ( -222.76  -154.18   -39.90     0.69)  ; 2
      )  ;  End of markers
       Incomplete
    |
      ( -269.77  -129.47   -22.57     0.92)  ; R-2-2, 1
      ( -268.17  -130.62   -24.75     0.92)  ; R-2-2, 2
      ( -266.79  -131.77   -26.13     0.92)  ; R-2-2, 3
       Incomplete
    )  ;  End of split
  )  ;  End of split
)
''')
    morph_fd = StringIO(string_section)
    sections = list(nasc._top_level_sections(morph_fd))
    expected = [['Dendrite'],
                 ['-290.87', '-113.09', '-16.32', '2.06'],  #id = 0, parent = -1
                 ['-290.87', '-113.09', '-16.32', '2.06'],
                 [['-277.14', '-119.13', '-18.02', '0.69'], #id = 1, parent = 0
                  ['-275.54', '-119.99', '-16.67', '0.69'],
                  'Normal',
                  '|',
                  ['-277.80', '-120.28', '-19.48', '0.92'], #id = 2, parent = 0
                  ['-276.65', '-121.14', '-20.20', '0.92'],
                  [['-267.94', '-128.61', '-22.57', '0.69'], #id = 3, parent = 2
                   ['-204.90', '-157.63', '-42.45', '0.69'],
                   'Incomplete',
                   '|',
                   ['-269.77', '-129.47', '-22.57', '0.92'], #id = 4, parent = 2
                   ['-268.17', '-130.62', '-24.75', '0.92'],
                   ['-266.79', '-131.77', '-26.13', '0.92'],
                   'Incomplete']]]
    eq_(sections[0], expected)

#what I think the
#https://developer.humanbrainproject.eu/docs/projects/morphology-documentation/0.0.2/h5v1.html
#would look like
MORPH_ASC = textwrap.dedent(
'''\
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
    (0 -10 0 2)
    (-3 -10 0 2)
 |
    (6 -10 0 2)
    (9 -10 0 2)
 )
)
''')

def test_read():
    try:
        fd, temp_file = tempfile.mkstemp('test_neurolucida')
        os.close(fd)
        with open(temp_file, 'w') as fd:
            fd.write(MORPH_ASC)
        rdw = nasc.read(temp_file)
        raw_data = rdw.data_block

        eq_(raw_data.shape, (19, 7))
        ok_(np.allclose(raw_data[:, COLS.ID], np.arange(0, 19)))  # correct ID
        # 3 is ID of end of the soma, 2 sections attach to this
        ok_(np.count_nonzero(raw_data[:, COLS.P] == 3),  2)
    finally:
        os.remove(temp_file)


def test_load_neurolucida_ascii():
    f = os.path.join(NEUROLUCIDA_PATH, 'sample.asc')
    ascii = io.load_data(f)
    ok_(isinstance(ascii, DataWrapper))
    eq_(len(ascii.data_block),
        4 + # soma
        9 + 2 + # axon + duplicate points for each sub-branch
        8 + 2)  # dendrite + duplicate points for each sub-branch
