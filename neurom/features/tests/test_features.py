import numpy as np
from neurom.features import make_iterable
from nose import tools as nt
from types import GeneratorType

def f(n): return (x for x in range(n))

def test_make_iterable_none():
	res = make_iterable(f, iterable_type=None)(5)
	nt.assert_true(isinstance(res, GeneratorType))

def test_make_iterable_numpy():
    
    res = make_iterable(f)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, np.ndarray))

def test_make_iterable_list():
    
    res = make_iterable(f, iterable_type=list)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, list))

def test_make_iterable_tuple():
    
    res = make_iterable(f, iterable_type=tuple)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, tuple))

@nt.raises(TypeError)
def test_make_iterable_other():
    def f(n): return (x for x in range(n))
    res = make_iterable(f, iterable_type=dict)(5)