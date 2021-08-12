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

"""Morphology features.

Any public function from this namespace can be called via the features mechanism. If calling
directly the function in this namespace can only accept a morphology as its input. If you want to
apply it to a morphology population then you must use the features mechanism e.g. ``features.get``.
The features mechanism does not allow you to apply these features to neurites.

>>> import neurom
>>> from neurom import features
>>> m = neurom.load_morphology('path/to/morphology')
>>> features.get('soma_surface_area', m)
>>> population = neurom.load_morphologies('path/to/morphs')
>>> features.get('sholl_crossings', population)

For more details see :ref:`features`.
"""


from functools import partial
import math
import numpy as np

from neurom import morphmath
from neurom.core.morphology import iter_neurites, iter_segments, Morphology
from neurom.core.types import tree_type_checker as is_type
from neurom.core.dataformat import COLS
from neurom.core.types import NeuriteType
from neurom.features import feature, NameSpace, neurite as nf

feature = partial(feature, namespace=NameSpace.NEURON)


@feature(shape=())
def soma_volume(morph):
    """Get the volume of a morphology's soma."""
    return morph.soma.volume


@feature(shape=())
def soma_surface_area(morph):
    """Get the surface area of a morphology's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    """
    return 4 * math.pi * morph.soma.radius ** 2


@feature(shape=())
def soma_radius(morph):
    """Get the radius of a morphology's soma."""
    return morph.soma.radius


@feature(shape=())
def max_radial_distance(morph, neurite_type=NeuriteType.all):
    """Get the maximum radial distances of the termination sections."""
    term_radial_distances = [nf.max_radial_distance(n)
                             for n in iter_neurites(morph, filt=is_type(neurite_type))]
    return max(term_radial_distances) if term_radial_distances else 0.


@feature(shape=(...,))
def number_of_sections_per_neurite(morph, neurite_type=NeuriteType.all):
    """List of numbers of sections per neurite."""
    return [nf.number_of_sections(n)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def total_length_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite lengths."""
    return [nf.total_length(n)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def total_area_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite areas."""
    return [nf.total_area(n)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def total_volume_per_neurite(morph, neurite_type=NeuriteType.all):
    """Neurite volumes."""
    return [nf.total_volume(n)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_origin_azimuths(morph, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin azimuths of a morph.

    The azimuth is defined as Angle between x-axis and the vector
    defined by (initial tree point - soma center) on the x-z plane.

    The range of the azimuth angle [-pi, pi] radians
    """
    def _azimuth(section, soma):
        """Azimuth of a section."""
        vector = morphmath.vector(section[0], soma.center)
        return np.arctan2(vector[COLS.Z], vector[COLS.X])

    return [_azimuth(n.root_node.points, morph.soma)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_origin_elevations(morph, neurite_type=NeuriteType.all):
    """Get a list of all the trunk origin elevations of a morph.

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

    return [_elevation(n.root_node.points, morph.soma)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_vectors(morph, neurite_type=NeuriteType.all):
    """Calculate the vectors between all the trunks of the morphology and the soma center."""
    return [morphmath.vector(n.root_node.points[0], morph.soma.center)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_angles(morph, neurite_type=NeuriteType.all):
    """Calculate the angles between all the trunks of the morph.

    The angles are defined on the x-y plane and the trees
    are sorted from the y axis and anticlock-wise.
    """
    vectors = np.array(trunk_vectors(morph, neurite_type=neurite_type))
    # In order to avoid the failure of the process in case the neurite_type does not exist
    if len(vectors) == 0:
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
def trunk_origin_radii(morph, neurite_type=NeuriteType.all):
    """Radii of the trunk sections of neurites in a morph."""
    return [n.root_node.points[0][COLS.R]
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def trunk_section_lengths(morph, neurite_type=NeuriteType.all):
    """List of lengths of trunk sections of neurites in a morph."""
    return [morphmath.section_length(n.root_node.points)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=())
def number_of_neurites(morph, neurite_type=NeuriteType.all):
    """Number of neurites in a morph."""
    return sum(1 for _ in iter_neurites(morph, filt=is_type(neurite_type)))


@feature(shape=(...,))
def neurite_volume_density(morph, neurite_type=NeuriteType.all):
    """Get volume density per neurite."""
    return [nf.volume_density(n)
            for n in iter_neurites(morph, filt=is_type(neurite_type))]


@feature(shape=(...,))
def sholl_crossings(morph, center=None, radii=None, neurite_type=NeuriteType.all):
    """Calculate crossings of neurites.

    Args:
        morph(Morphology|list): morphology or a list of neurites
        center(Point): center point, if None then soma center is taken
        radii(iterable of floats): radii for which crossings will be counted,
            if None then soma radius is taken
        neurite_type(NeuriteType): Type of neurite to use. By default ``NeuriteType.all`` is used.

    Returns:
        Array of same length as radii, with a count of the number of crossings
        for the respective radius

    This function can also be used with a list of sections, as follow::

        secs = (sec for sec in nm.iter_sections(morph) if complex_filter(sec))
        sholl = nm.features.neuritefunc.sholl_crossings(secs,
                                                        center=morph.soma.center,
                                                        radii=np.arange(0, 1000, 100))
    """
    def _count_crossings(neurite, radius):
        """Used to count_crossings of segments in neurite with radius."""
        r2 = radius ** 2
        count = 0
        for start, end in iter_segments(neurite):
            start_dist2, end_dist2 = (morphmath.point_dist2(center, start),
                                      morphmath.point_dist2(center, end))

            count += int(start_dist2 <= r2 <= end_dist2 or
                         end_dist2 <= r2 <= start_dist2)

        return count

    if center is None or radii is None:
        assert isinstance(morph, Morphology) and morph.soma, \
            '`sholl_crossings` input error. If `center` or `radii` is not set then `morph` is ' \
            'expected to be an instance of Morphology and have a soma.'
        if center is None:
            center = morph.soma.center
        if radii is None:
            radii = [morph.soma.radius]
    return [sum(_count_crossings(neurite, r)
                for neurite in iter_neurites(morph, filt=is_type(neurite_type)))
            for r in radii]


@feature(shape=(...,))
def sholl_frequency(morph, neurite_type=NeuriteType.all, step_size=10, bins=None):
    """Perform Sholl frequency calculations on a morph.

    Args:
        morph(Morphology): a morphology
        neurite_type(NeuriteType): which neurites to operate on
        step_size(float): step size between Sholl radii
        bins(iterable of floats): custom binning to use for the Sholl radii. If None, it uses
        intervals of step_size between min and max radii of ``morphologies``.

    Note:
        Given a morphology, the soma center is used for the concentric circles,
        which range from the soma radii, and the maximum radial distance
        in steps of `step_size`. Each segment of the morphology is tested, so a neurite that
        bends back on itself, and crosses the same Sholl radius will get counted as
        having crossed multiple times.
    """
    neurite_filter = is_type(neurite_type)

    if bins is None:
        min_soma_edge = morph.soma.radius
        max_radii = max(np.max(np.linalg.norm(n.points[:, COLS.XYZ], axis=1))
                        for n in morph.neurites if neurite_filter(n))
        bins = np.arange(min_soma_edge, min_soma_edge + max_radii, step_size)

    return sholl_crossings(morph, morph.soma.center, bins, neurite_type)
