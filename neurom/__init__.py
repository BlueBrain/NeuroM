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

'''NeuroM neurom morphology analysis package

Examples:

    Load a neuron

    >>> import neurom as nm
    >>> nrn = nm.load_neuron('some/data/path/morph_file.swc')

    Obtain some morphometrics using the get function

    >>> ap_seg_len = nm.get('segment_lengths', nrn, neurite_type=nm.APICAL_DENDRITE)
    >>> ax_sec_len = nm.get('section_lengths', nrn, neurite_type=nm.AXON)

    Load neurons from a directory. This loads all SWC, HDF5 or NeuroLucida .asc\
    files it finds and returns a list of neurons

    >>> import numpy as np  # For mean value calculation
    >>> nrns = nm.load_neurons('some/data/directory')
    >>> for nrn in nrns:
    ...     print 'mean section length', np.mean(nm.get('section_lengths', nrn))

    Apply a function to a selection of neurites in a neuron or population.
    This example gets the number of points in each axon in a neuron population

    >>> import neurom as nm
    >>> filter = lambda n : n.type == nm.AXON
    >>> mapping = lambda n : len(n.points)
    >>> n_points = [n for n in nm.iter_neurites(nrns, mapping, filter)]

'''

from .version import VERSION as __version__
from .core import iter_neurites, iter_sections
from .fst import load_neuron, load_neurons, NeuriteType, get, NEURITE_TYPES


APICAL_DENDRITE = NeuriteType.apical_dendrite
BASAL_DENDRITE = NeuriteType.basal_dendrite
AXON = NeuriteType.axon
SOMA = NeuriteType.soma
ANY_NEURITE = NeuriteType.all
