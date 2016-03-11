import numpy as np
from neurom.features import make_iterable
from nose import tools as nt
from types import GeneratorType

def f(n): return (x for x in range(n))

def test_make_iterable_numpy():
    
    res = make_iterable()(f)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, np.ndarray))

def test_make_iterable_list():
    
    res = make_iterable(iterable_type=list)(f)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, list))

def test_make_iterable_tuple():
    
    res = make_iterable(iterable_type=tuple)(f)(5)
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
    nt.assert_true(isinstance(res, tuple))

@nt.raises(TypeError)
def test_make_iterable_other():
    def f(n): return (x for x in range(n))
    res = make_iterable(iterable_type=dict)(f)(5)