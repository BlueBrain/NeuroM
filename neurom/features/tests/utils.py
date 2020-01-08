import numpy as np
from nose import tools as nt


def _close(a, b, debug=False, rtol=1e-05, atol=1e-08):
    a, b = list(a), list(b)
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
        print('\na - b:%s\n' % (a - b))
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.allclose(a, b, rtol=rtol, atol=atol))


def _equal(a, b, debug=False):
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
    nt.assert_equal(len(a), len(b))
    nt.assert_true(np.alltrue(a == b))
