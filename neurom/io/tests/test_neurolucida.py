import os
import textwrap
from io import StringIO

import numpy as np
from mock import patch
from nose.tools import eq_, ok_

import neurom as nm
from neurom import load_neuron
import neurom.io as io
from neurom.core.dataformat import COLS
from numpy.testing import assert_array_equal
from nose import tools as nt

from neurom.exceptions import RawDataError, SomaError

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')
NEUROLUCIDA_PATH = os.path.join(DATA_PATH, 'neurolucida')




def test_soma():
    string_section = u'''
                         ("CellBody"
                         (Color Red)
                         (CellBody)
                         (1 1 0 1 S1)
                         (-1 1 0 1 S2)
                         (-1 -1 0 2 S3)
                         )
        '''
    n = nm.load_neuron(('asc', string_section))

    assert_array_equal(n.soma.points,
                       [[1,1,0, 0.5],
                        [-1,1,0, 0.5],
                        [-1,-1,0,1]])

    nt.assert_equal(len(n.neurites), 0)

@nt.raises(SomaError)
def test_multiple_soma():
    string_section = u'''
                         ("CellBody"
                         (Color Red)
                         (CellBody)
                         (1 1 0 1 S1)
                         (-1 1 0 1 S2)
                         (-1 -1 0 2 S3)
                         )

                        ("CellBody"
                         (Color Red)
                         (CellBody)
                         (1 1 0 1 S1)
                         (-1 1 0 1 S2)
                         (-1 -1 0 2 S3)
                         )
        '''
    load_neuron(('asc', string_section))

def test_single_neurite_no_soma():
    string_section = u'''

                         ( (Color Yellow)
                         (Axon)
                         (Set "axons")

    ;; An commented line and some empty lines

                         (  1.2  2.7   1.0     13)  ;; Some comment
                         (  1.2  3.7   2.0     13)

                         Generated
                         )  ;  End of tree'''
    n = nm.load_neuron(('asc', string_section))

    assert_array_equal(n.soma.points, np.empty((0,4)))
    nt.assert_equal(len(n.neurites), 1)
    assert_array_equal(n.neurites[0].points,
                       np.array([[1.2, 2.7, 1.0, 6.5],
                                 [1.2, 3.7, 2.0, 6.5]], dtype=np.float32))


def test_skip_header():
    '''Test that the header does not cause any issue'''
    str_neuron = '''(FilledCircle
                         (Color RGB (64, 0, 128))
                         (Name "Marker 11")
                         (Set "axons")
                         ( -189.59    55.67    28.68     0.12)  ; 1
                         )  ;  End of markers

                         ((Color Yellow)
                         (Axon)
                         (Set "axons")
                         (  1.2  2.7   1.0     13)
                         (  1.2  3.7   2.0     13)
                         )'''

    n = nm.load_neuron(('asc', str_neuron))
    nt.assert_equal(len(n.neurites), 1)
    assert_array_equal(n.neurites[0].points,
                       np.array([[1.2, 2.7, 1.0, 6.5],
                                 [1.2, 3.7, 2.0, 6.5]], dtype=np.float32))





without_duplicate = '''
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
                     '''

with_duplicate = '''
                     ((Dendrite)
                      (3 -4 0 2)
                      (3 -6 0 2)
                      (3 -8 0 2)
                      (3 -10 0 2)
                      (
                        (3 -10 0 2) ; duplicate
                        (0 -10 0 2)
                        (-3 -10 0 2)
                        |
                        (3 -10 0 2) ; duplicate
                        (6 -10 0 2)
                        (9 -10 0 2)
                      )
                      )
                     '''

def test_read_with_duplicates():
    '''Section points are duplicated in the file'''
# what I think the
# https://developer.humanbrainproject.eu/docs/projects/morphology-documentation/0.0.2/h5v1.html
# would look like
    n = load_neuron(('asc',
                     with_duplicate))

    nt.assert_equal(len(n.neurites), 1)

    assert_array_equal(n.neurites[0].points,

                       # Duplicate points are not present
                       [[3, -4, 0, 1],
                        [3, -6, 0, 1],
                        [3, -8, 0, 1],
                        [3, -10, 0, 1],
                        [0, -10, 0, 1],
                        [-3, -10, 0, 1],
                        [6, -10, 0, 1],
                        [9, -10, 0, 1]])

    assert_array_equal(n.neurites[0].root_node.points,
                       [[3, -4, 0, 1],
                        [3, -6, 0, 1],
                        [3, -8, 0, 1],
                        [3, -10, 0, 1]])


    assert_array_equal(n.neurites[0].root_node.children[0].points,
                       [[3, -10, 0, 1],
                        [0, -10, 0, 1],
                        [-3, -10, 0, 1]])

    assert_array_equal(n.neurites[0].root_node.children[1].points,
                       [[3, -10, 0, 1],
                        [6, -10, 0, 1],
                        [9, -10, 0, 1]])


def test_read_without_duplicates():
    n_with_duplicate = load_neuron(('asc', with_duplicate))
    n_without_duplicate = load_neuron(('asc', without_duplicate))

    assert_array_equal(n_with_duplicate.neurites[0].root_node.children[0].points,
                       n_without_duplicate.neurites[0].root_node.children[0].points)


    assert_array_equal(n_with_duplicate.neurites[0].points,
                       n_without_duplicate.neurites[0].points)


@nt.raises(RawDataError)
def test_broken_duplicate():
    load_neuron(('asc',
                 '''
                     ((Dendrite)
                      (3 -4 0 2)
                      (3 -6 0 2)
                      (3 -8 0 2)
                      (3 -10 0 2)
                      (
                        (3 -10 0 40) ; <-- duplicate with different radii
                        (0 -10 0 2)
                        (-3 -10 0 2)
                        |
                        (3 -10 0 2) ; <-- good duplicate
                        (6 -10 0 2)
                        (9 -10 0 2)
                      )
                      )
                     '''))


@nt.raises(RawDataError)
def test_unfinished_file():
    load_neuron(('asc',
                 '''
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
'''))

def test_empty_sibling():
    n = load_neuron(('asc',
                       '''
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
                       )
                      )
                 '''))


    assert_array_equal(n.neurites[0].points,
                         np.array([[3, -4, 0, 1],
                                   [3, -6, 0, 1],
                                   [3, -8, 0, 1],
                                   [3, -10, 0, 1],
                                   [0, -10, 0, 1],
                                   [-3, -10, 0, 1]],
                                  dtype=np.float32))


def test_single_children():
    n = load_neuron(('asc',
                 '''
                     ((Dendrite)
                      (3 -4 0 2)
                      (3 -6 0 2)
                      (3 -8 0 2)
                      (3 -10 0 2)
                      (
                        (3 -10 0 2)
                        (0 -10 0 2)
                        (-3 -10 0 2)
                       )
                      )
                 '''))

    assert_array_equal(n.neurites[0].points,
                         np.array([[3, -4, 0, 1],
                                   [3, -6, 0, 1],
                                   [3, -8, 0, 1],
                                   [3, -10, 0, 1],
                                   [0, -10, 0, 1],
                                   [-3, -10, 0, 1]],
                                  dtype=np.float32))

def test_markers():
    '''Test that markers do not prevent file from being read correctly'''
    n = load_neuron(('asc',
                 '''
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
'''))

    nt.assert_equal(len(n.neurites), 1)

    res = np.array([[ -290.87,  -113.09,   -16.32,     1.03],
                    [ -290.87,  -113.09,   -16.32,     1.03],
                    [ -277.14,  -119.13,   -18.02,     0.345],
                    [ -275.54,  -119.99,   -16.67,     0.345],
                    [ -277.80,  -120.28,   -19.48,     0.46],
                    [ -276.65,  -121.14,   -20.20,     0.46],
                    [ -267.94,  -128.61,   -22.57,     0.345],
                    [ -204.90,  -157.63,   -42.45,     0.345],
                    [ -269.77,  -129.47,   -22.57,     0.46],
                    [ -268.17,  -130.62,   -24.75,     0.46],
                    [ -266.79,  -131.77,   -26.13,     0.46]],
                   dtype=np.float32)

    assert_array_equal(n.neurites[0].points,
                       res)
