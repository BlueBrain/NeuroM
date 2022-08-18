# Copyright (c) 2020, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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

"""Population features.

Any public function from this namespace can be called via features mechanism. Functions in this
namespace can only accept a morphology population as its input no matter how called.

>>> import neurom
>>> from neurom import features
>>> pop = neurom.load_morphologies('path/to/morphs')
>>> features.get('sholl_frequency', pop)

For more details see :ref:`features`.
"""


from functools import partial
import numpy as np

from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType
from neurom.core.types import tree_type_checker as is_type
from neurom.features import feature, NameSpace
from neurom.features.morphology import sholl_crossings

feature = partial(feature, namespace=NameSpace.POPULATION)


@feature(shape=(...,))
def sholl_frequency(morphs, neurite_type=NeuriteType.all, step_size=10, bins=None):
    """Perform Sholl frequency calculations on a population of morphs.

    Args:
        morphs(list|Population): list of morphologies or morphology population
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii
        bins(iterable of floats): custom binning to use for the Sholl radii. If None, it uses
        intervals of step_size between min and max radii of ``morphs``.

    Note:
        Given a population, the concentric circles range from the smallest soma radius to the
        largest radial neurite distance in steps of `step_size`. Each segment of the morphology is
        tested, so a neurite that bends back on itself, and crosses the same Sholl radius will
        get counted as having crossed multiple times.
    """
    neurite_filter = is_type(neurite_type)

    if bins is None:
        min_soma_edge = min(n.soma.radius for n in morphs)
        max_radii = max(np.max(np.linalg.norm(n.points[:, COLS.XYZ], axis=1))
                        for m in morphs
                        for n in m.neurites if neurite_filter(n))
        bins = np.arange(min_soma_edge, min_soma_edge + max_radii, step_size)

    return np.array([
        sholl_crossings(m, neurite_type, m.soma.center, bins)
        for m in morphs
    ]).sum(axis=0)
