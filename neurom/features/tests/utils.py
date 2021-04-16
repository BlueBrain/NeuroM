import numpy as np


def _close(a, b, debug=False, rtol=1e-05, atol=1e-08):
    a, b = list(a), list(b)
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
        print('\na - b:%s\n' % (a - b))
    assert len(a) == len(b)
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def _equal(a, b, debug=False):
    if debug:
        print('\na.shape: %s\nb.shape: %s\n' % (a.shape, b.shape))
        print('\na: %s\nb:%s\n' % (a, b))
    assert len(a) == len(b)
    assert np.alltrue(a == b)
