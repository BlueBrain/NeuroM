'''Data format definitions'''


class Rows(object):
    '''Row labels for internal data representation

    These mirror the SWC data format.
    '''
    (ID, TYPE, X, Y, Z, R, P) = xrange(7)


ROOT_ID = -1
