# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

''' Module for morphology data loading and access

Data is unpacked into a 2-dimensional raw data block:

    [X, Y, Z, R, TYPE, ID, PARENT_ID]

There is one such row per measured point.

Functions to umpack the data and a higher level wrapper are provided. See

* load_data
'''
import os


def load_data(filename):
    '''Unpack filename and return a RawDataWrapper object containing the data

    Determines format from extension. Currently supported:

        * SWC (case-insensitive extension ".swc")
        * H5 v1 and v2 (case-insensitive extension ".h5"). Attempts to
          determine the version from the contents of the file
        * Neurolucida ASCII (case-insensitive extension ".asc")
    '''

    def read_h5(filename):
        '''Lazy loading of HDF5 reader'''
        from .hdf5 import H5
        return H5.read(filename)

    def read_swc(filename):
        '''Lazy loading of SWC reader'''
        from .swc import SWC
        return SWC.read(filename)

    def read_neurolucida(filename):
        '''Lazy loading of Neurolucida ASCII reader'''
        from .neurolucida import NeurolucidaASC
        return NeurolucidaASC.read(filename)

    _READERS = {
        'swc': read_swc,
        'h5': read_h5,
        'asc': read_neurolucida,
    }
    extension = os.path.splitext(filename)[1][1:]
    return _READERS[extension.lower()](filename)
