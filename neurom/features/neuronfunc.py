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

"""Neuron features.

Any public function from this namespace can be called via features mechanism. The public
function in this namespace can only accept a neuron as its input. If you want to apply it to
a neuron population then you must use the features mechanism e.g. `features.get`. Even via
features mechanism the function can't be applied to a single neurite.

>>> import neurom
>>> from neurom import features
>>> nrn = neurom.load_neuron('path/to/neuron')
>>> features.get('soma_surface_area', nrn)
>>> population = neurom.load_neurons('path/to/neurons')
>>> features.get('sholl_crossings', population)
"""


from functools import partial
import math
import numpy as np

from neurom import morphmath
from neurom.core.neuron import iter_neurites
from neurom.core.types import tree_type_checker as is_type
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType
from neurom.features import feature, NameSpace, neuritefunc
from neurom.features.sectionfunc import sholl_crossings

feature = partial(feature, namespace=NameSpace.NEURON)


@feature(shape=())
def soma_volume(neuron):
    """Get the volume of a neuron's soma."""
    return neuron.soma.volume


@feature(shape=())
def soma_surface_area(neuron):
    """Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    """
    return 4 * math.pi * neuron.soma.radius ** 2


@feature(shape=())
def soma_radius(neuron):
    """Get the radius of a neuron's soma."""
    return neuron.soma.radius


@feature(shape=())
def max_radial_distance(neuron, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections.
     TODO Put it as an example to the doc page"""
    term_radial_distances = [neuritefunc.max_radial_distance(s)
                             for s in iter_neurites(neuron, filt=is_type(neurite_type))]
    return max(term_radial_distances) if term_radial_distances else 0.


@feature(shape=(...,))
def number_of_sections_per_neurite(neuron, neurite_type=NeuriteType.all):
    """Neurite lengths."""
    return [neuritefunc.number_of_sections(s)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=(...,))
def neurite_lengths(neuron, neurite_type=NeuriteType.all):
    """Neurite lengths."""
    return [neuritefunc.total_length(s)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=())
def neurite_volumes(neuron, neurite_type=NeuriteType.all):
    """Get the volume."""
    return [neuritefunc.total_volume(s)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_origin_azimuths(neuron, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin azimuths of a neuron.

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    """
    def _azimuth(section, soma):
        """Azimuth of a section."""
        vector = morphmath.vector(section[0], soma.center)
        return np.arctan2(vector[COLS.Z], vector[COLS.X])

    return [_azimuth(s.root_node.points, neuron.soma)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_origin_elevations(neuron, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin elevations of a neuron.

    The elevation is defined as the angle between x-axis and the
    vector defined by (initial tree point - soma center)
    on the x-y half-plane.

    The range of the elevation angle [-pi/2, pi/2] radians
    """
    def _elevation(section, soma):
        """Elevation of a section."""
        vector = morphmath.vector(section[0], soma.center)
        norm_vector = np.linalg.norm(vector)

        if norm_vector >= np.finfo(type(norm_vector)).eps:
            return np.arcsin(vector[COLS.Y] / norm_vector)
        raise ValueError("Norm of vector between soma center and section is almost zero.")

    return [_elevation(s.root_node.points, neuron.soma)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_vectors(neuron, neurite_type=NeuriteType.all):
    """Calculates the vectors between all the trunks of the neuron and the soma center."""
    return np.array([morphmath.vector(s.root_node.points[0], neuron.soma.center)
                     for s in iter_neurites(neuron, filt=is_type(neurite_type))])


@feature(shape=(...,))
def trunk_angles(neuron, neurite_type=NeuriteType.all):
    """Calculates the angles between all the trunks of the neuron.

    The angles are defined on the x-y plane and the trees
    are sorted from the y axis and anticlock-wise.
    """
    vectors = trunk_vectors(neuron, neurite_type=neurite_type)
    # In order to avoid the failure of the process in case the neurite_type does not exist
    if not vectors.size:
        return []

    def _sort_angle(p1, p2):
        """Angle between p1-p2 to sort vectors."""
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return ang1 - ang2

    # Sorting angles according to x-y plane
    order = np.argsort(np.array([_sort_angle(i / np.linalg.norm(i), [0, 1])
                                 for i in vectors[:, 0:2]]))

    ordered_vectors = vectors[order][:, [COLS.X, COLS.Y]]

    return [morphmath.angle_between_vectors(ordered_vectors[i], ordered_vectors[i - 1])
            for i, _ in enumerate(ordered_vectors)]


@feature(shape=(...,))
def trunk_origin_radii(neuron, neurite_type=NeuriteType.all):
    """Radii of the trunk sections of neurites in a neuron."""
    return [s.root_node.points[0][COLS.R]
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_section_lengths(neuron, neurite_type=NeuriteType.all):
    """List of lengths of trunk sections of neurites in a neuron."""
    return [morphmath.section_length(s.root_node.points)
            for s in iter_neurites(neuron, filt=is_type(neurite_type))]


@feature(shape=())
def number_of_neurites(neuron, neurite_type=NeuriteType.all):
    """Number of neurites in a neuron."""
    return sum(1 for _ in iter_neurites(neuron, filt=is_type(neurite_type)))


@feature(shape=(...,))
def sholl_frequency(neuron, neurite_type=NeuriteType.all, step_size=10, bins=None):
    """Perform Sholl frequency calculations on a neuron.

    Args:
        neuron(Neuron): a neuron
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii
        bins(iterable of floats): custom binning to use for the Sholl radii. If None, it uses
        intervals of step_size between min and max radii of ``neurons``.

    Note:
        Given a neuron, the soma center is used for the concentric circles,
        which range from the soma radii, and the maximum radial distance
        in steps of `step_size`. Each segment of the neuron is tested, so a neurite that
        bends back on itself, and crosses the same Sholl radius will get counted as
        having crossed multiple times.
    """
    neurite_filter = is_type(neurite_type)

    if bins is None:
        min_soma_edge = neuron.soma.radius
        max_radii = max(np.max(np.linalg.norm(s.points[:, COLS.XYZ], axis=1))
                        for s in neuron.neurites if neurite_filter(s))
        bins = np.arange(min_soma_edge, min_soma_edge + max_radii, step_size)

    return sholl_crossings(neuron, neuron.soma.center, bins, neurite_type)
