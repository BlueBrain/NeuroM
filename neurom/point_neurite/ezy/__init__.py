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

''' Deprecated module for quick and easy neuron morphology analysis tools

Note:
    This module is deprecated. Do not use it for anything.

Examples:

    Load a neuron

    >>> from neurom import ezy
    >>> nrn = ezy.load_neuron('some/data/path/morph_file.swc')

    Obtain some morphometrics

    >>> ap_seg_len = ezy.get('segment_lengths', nrn, neurite_type=ezy.NeuriteType.apical_dendrite)
    >>> ax_sec_len = ezy.get('section_lengths', nrn, neurite_type=ezy.NeuriteType.axon)

    View it in 2D and 3D

    >>> fig2d, ax2d = ezy.view(nrn)
    >>> fig2d.show()
    >>> fig3d, ax3d = ezy.view3d(nrn)
    >>> fig3d.show()

    Load neurons from a directory. This loads all SWC or HDF5 files it finds\
    and returns a list of neurons

    >>> import numpy as np  # For mean value calculation
    >>> nrns = ezy.load_neurons('some/data/directory')
    >>> for nrn in nrns:
    ...     print 'mean section length', np.mean(ezy.get('section_lengths', nrn))

'''
from functools import partial, update_wrapper
from neurom.point_neurite.core import Neuron
from neurom.core.population import Population
from neurom.core.types import NeuriteType, NEURITES as NEURITE_TYPES
from neurom.view.view import neuron as _view
from neurom.view.view import neuron3d as _view3d
from neurom.io.utils import get_morph_files
from neurom.point_neurite.features import get
from neurom.point_neurite.io import load_neuron, load_neurons
from neurom.utils import deprecated, deprecated_module


deprecated_module(__name__)

TreeType = NeuriteType  # For backwards compatibility

load_population = load_neurons

view = deprecated(msg='Use neurom.viewer.draw instead.',
                  fun_name='%s.view' % __name__)(_view)

view3d = deprecated(msg='Use neurom.viewer.draw instead.',
                    fun_name='%s.view3d' % __name__)(_view)
