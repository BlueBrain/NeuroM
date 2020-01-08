from neurom import load_neuron
from neurom.apps.annotate import annotate, generate_annotation
from neurom.check import CheckResult
from neurom.check.neuron_checks import has_no_narrow_start, has_no_single_children
from nose import tools as nt


def test_generate_annotation():
    checker_ok = CheckResult(True)
    checker_not_ok = CheckResult(False, [('section 1', [[1, 2, 3], [4, 5, 6]]),
                                         ('section 2', [[7, 8, 9], [10, 11, 12]])])

    settings = {'color': 'blue', 'label': 'circle', 'name': 'dangling'}
    nt.assert_equal(generate_annotation(checker_ok, settings), "")

    correct_result = '''

(circle   ; MUK_ANNOTATION
    (Color blue)   ; MUK_ANNOTATION
    (Name "dangling")   ; MUK_ANNOTATION
    (      1.00       2.00       3.00 0.50)   ; MUK_ANNOTATION
    (      4.00       5.00       6.00 0.50)   ; MUK_ANNOTATION
    (      7.00       8.00       9.00 0.50)   ; MUK_ANNOTATION
    (     10.00      11.00      12.00 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
'''

    c = generate_annotation(checker_not_ok, settings)
    nt.assert_equal(generate_annotation(checker_not_ok, settings), correct_result)


def test_annotate():

    correct_result = """

(Circle1   ; MUK_ANNOTATION
    (Color Blue)   ; MUK_ANNOTATION
    (Name "narrow start")   ; MUK_ANNOTATION
    (      0.00       0.00       2.00 0.50)   ; MUK_ANNOTATION
)   ; MUK_ANNOTATION
"""

    checkers = {has_no_narrow_start: {"name": "narrow start",
                                      "label": "Circle1",
                                      "color": "Blue"}}

    neuron = load_neuron('test_data/swc/narrow_start.swc')
    results = [checker(neuron) for checker in checkers.keys()]
    nt.assert_equal(annotate(results, checkers.values()), correct_result)


# def test_single_children():

#     checkers = {has_no_single_children: {"name": "single child",
#                                          "label": "Circle6",
#                                          "color": "Red"}}

#     neuron = load_neuron(('asc', """
# ( (Color Blue)
#   (Axon)
#   (0 5 0 2)
#   (2 9 0 2)
#   (0 13 0 2)
#   (
#     (2 13 0 2)
#     (4 13 0 2)
#     (6 13 0 2)
#   )
# )
# """))

#     results = [checker(neuron) for checker in checkers.keys()]
#     nt.assert_equal(annotate(results, checkers.values()),
#                     """

# (Circle6   ; MUK_ANNOTATION
#     (Color Red)   ; MUK_ANNOTATION
#     (Name "single child")   ; MUK_ANNOTATION
#     (0.0 13.0 0.0 0.50)   ; MUK_ANNOTATION
# )   ; MUK_ANNOTATION
# """)
