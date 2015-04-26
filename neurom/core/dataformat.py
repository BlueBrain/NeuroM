'''Data format definitions'''


class COLS(object):
    '''Column labels for internal data representation

    These mirror the SWC data format.
    '''
    (ID, TYPE, X, Y, Z, R, P) = xrange(7)


class POINT_TYPE(object):
    '''Point types.

    These follow SWC specification.
    '''
    (UNDEFINED, SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE,
     FORK_POINT, END_POINT, CUSTOM) = xrange(8)

    NEURITES = (AXON, BASAL_DENDRITE, APICAL_DENDRITE)


ROOT_ID = -1
