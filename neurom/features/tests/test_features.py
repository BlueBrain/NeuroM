import numpy as np
from neurom.features import _pkg
from nose import tools as nt

def test_pkg():
    def f(): return (x for x in range(5))
    res = _pkg(f)()
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))
