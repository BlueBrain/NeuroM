from neurom import load_neuron
from neurom.apps.annotate import annotate, generate_annotation
from neurom.check import CheckResult
from neurom.check.neuron_checks import has_no_narrow_start
from nose import tools as nt


def test_generate_annotation():
    checker_ok = CheckResult(True)
    checker_not_ok = CheckResult(False, [('section 1', [[1, 2, 3], [4, 5, 6]]),
                                         ('section 2', [[7, 8, 9], [10, 11, 12]])])

    settings = {'color': 'blue', 'label': 'circle', 'name': 'dangling'}
    nt.assert_equal(generate_annotation(checker_ok, settings), "")

    correct_result = """

(circle   ; MUK_ANNOTATION
    (Color blue)   ; MUK_ANNOTATION
    (Name "dangling")   ; MUK_ANNOTATION
    (1 2 3 0.50)   ; MUK_ANNOTATION
    (4 5 6 0.50)   ; MUK_ANNOTATION
    (7 8 9 0.50)   ; MUK_ANNOTATION
    (10 11 12 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
"""

    nt.assert_equal(generate_annotation(checker_not_ok, settings), correct_result)


def test_annotate():

    correct_result = """

(Circle1   ; MUK_ANNOTATION
    (Color Blue)   ; MUK_ANNOTATION
    (Name "narrow start")   ; MUK_ANNOTATION
    (0.0 0.0 0.0 0.50)   ; MUK_ANNOTATION
    (0.0 0.0 0.0 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
"""

    checkers = {has_no_narrow_start: {"name": "narrow start",
                                      "label": "Circle1",
                                      "color": "Blue"}}

    neuron = load_neuron('test_data/swc/Neuron_zero_radius.swc')
    results = [checker(neuron) for checker in checkers.keys()]
    nt.assert_equal(annotate(results, checkers.values()), correct_result)
