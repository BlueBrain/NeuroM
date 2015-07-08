'''HDF5 reader'''
from itertools import izip
import h5py
import numpy as np


class H5V1(object):
    '''Read HDF5 v1 files and unpack into internal raw data block

    Input row format:
        points: [X, Y, Z, D] (ID is position)
        groups: [FIRST_POINT_ID, TYPE, PARENT_GROUP_ID]

    Internal row format: [X, Y, Z, R, TYPE, ID, PARENT_ID]
    '''

    @staticmethod
    def read(filename):
        '''Read an HDF5 v1 file and return a tuple of data, offset, format.'''
        data = H5V1.unpack_data(h5py.File(filename))
        offset = 0 # H5V1 is index based, so there's no offset
        return data, offset, 'H5V1'

    @staticmethod
    def unpack_data(h5file):
        '''Unpack data from h5 ve file into internal format'''
        points = np.array(h5file['points'])
        groups = np.array(h5file['structure'])

        return np.array([np.array([p[0], p[1], p[2], p[3] / 2.,
                                   find_group(i, groups)[1], i,
                                   find_parent_id(i, groups)])
                         for i, p in enumerate(points)])


def find_group(point_id, groups):
    '''Find the structure group a points id belongs to

    Return: group or section point_id belongs to. Last group if
            point_id out of bounds.
    '''
    return next((start for (start, end) in izip(groups, groups[1:])
                 if point_id >= start[0] and point_id < end[0]), groups[-1])


def find_parent_id(point_id, groups):
    '''Find the parent ID of a point'''
    group = find_group(point_id, groups)
    if point_id != group[0]:
        # point is not first point in section
        # so parent is previous point
        return point_id - 1
    else:
        # parent is last point in parent group
        parent_group_id = group[2]
        # get last point in parent group
        return groups[parent_group_id + 1][0] - 1


if __name__ == '__main__':

    FILENAME = 'test_data/h5/v1/Neuron_2_branch.h5'

    _data, _, _fmt = H5V1.read(FILENAME)

    _h5file = h5py.File(FILENAME)
    _points = np.array(_h5file['points'])
    _groups = np.array(_h5file['structure'])
