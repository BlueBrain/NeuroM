from nose import tools as nt

from neurom import load_neuron
from neurom.apps.annotate import *
from neurom.check import CheckResult


def test_generate_annotation():
    def checker_ok(*args):
        return CheckResult(True)

    def checker_not_ok(*args):
        return CheckResult(False, [('section 1', [[1, 2, 3], [4, 5, 6]]),
                                   ('section 2', [[7, 8, 9], [10, 11, 12]])])

    settings = {'color': 'blue', 'label': 'circle', 'name': 'dangling'}
    nt.assert_equal(generate_annotation(None, (checker_ok, settings)), "")

    correct_result = '''

(circle   ; MUK_ANNOTATION
    (Color blue)   ; MUK_ANNOTATION
    (Name "dangling")   ; MUK_ANNOTATION
    (1 2 3 0.50)   ; MUK_ANNOTATION
    (4 5 6 0.50)   ; MUK_ANNOTATION
    (7 8 9 0.50)   ; MUK_ANNOTATION
    (10 11 12 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
'''

    nt.assert_equal(generate_annotation(None, (checker_not_ok, settings)), correct_result)


def test_annotate():

    correct_result = """

(Circle1   ; MUK_ANNOTATION
    (Color Blue)   ; MUK_ANNOTATION
    (Name "narrow start")   ; MUK_ANNOTATION
    (0.0 0.0 0.0 0.50)   ; MUK_ANNOTATION
    (0.0 0.0 0.0 0.50)   ; MUK_ANNOTATION
    (0.0 0.0 0.0 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
"""

    nt.assert_equal(annotate(load_neuron(
        'test_data/swc/Neuron_zero_radius.swc')), correct_result)
