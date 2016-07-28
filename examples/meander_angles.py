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

'''Calculate inter-segment angles'''

import logging
import numpy as np
from neurom.core.dataformat import COLS
from neurom.point_neurite import triplets as trip
from neurom.point_neurite.io.utils import load_data, make_neuron
from neurom import iter_neurites


# root level logger. This would be a top level application logger.
logging.basicConfig()
LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG.handlers[0].setFormatter(fmt)


if __name__ == '__main__':

    filename = 'test_data/swc/Neuron.swc'
    rd = load_data(filename)
    nrn = make_neuron(rd)

    for tt in nrn.neurites:
        print 'Tree ID: {0}, type: {1}'.format(tt.value[COLS.ID], tt.value[COLS.TYPE])
        for a in iter_neurites(tt, trip.meander_angle):
            LOG.debug('Angle %f', a)
            if np.isnan(a):
                LOG.warn('Found NaN angle. Check for zero length segments!')
