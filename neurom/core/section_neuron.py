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

'''Section tree module'''

import math
from collections import namedtuple
import numpy as np
from neurom.io.hdf5 import H5
from neurom.core.types import NeuriteType
from neurom.core.tree import Tree, ipreorder, ibifurcation_point
from neurom.core.types import tree_type_checker as is_type
from neurom.core.dataformat import POINT_TYPE
from neurom.core.dataformat import COLS
from neurom.core.tree import i_chain2, iupstream
from neurom.core.neuron import make_soma
from neurom.analysis import morphmath as mm


class SEC(object):
    '''Enum class with section content indices'''
    (START, END, TYPE, ID, PID) = xrange(5)


Neuron = namedtuple('Neuron', 'soma, neurites, data_block')


class SecDataWrapper(object):
    '''Class holding a raw data block and section information'''

    def __init__(self, data_block, fmt, sections):
        '''Section Data Wrapper'''
        self.data_block = data_block
        self.fmt = fmt
        self.sections = sections

    def neurite_trunks(self):
        '''Get the section IDs of the intitial neurite sections'''
        sec = self.sections
        return [ss[SEC.ID] for ss in sec
                if sec[ss[SEC.PID]][SEC.TYPE] == POINT_TYPE.SOMA and
                ss[SEC.TYPE] != POINT_TYPE.SOMA]

    def soma_points(self):
        '''Get the soma points'''
        db = self.data_block
        return db[db[:, COLS.TYPE] == POINT_TYPE.SOMA]


def set_neurite_type(tree):
    '''Calculate and set the neurite type of a tree'''
    tree_types = tuple(NeuriteType)
    types = np.array([node.value[0][COLS.TYPE] for node in ipreorder(tree)])
    tree.type = tree_types[int(np.median(types))]


def make_tree(rdw, start_node=0, post_action=None):
    '''Build a section tree'''
    # One pass over sections to build nodes
    nodes = [Tree(rdw.data_block[sec[SEC.START]: sec[SEC.END]])
             for sec in rdw.sections[start_node:]]

    # One pass over nodes to connect children to parents
    for i in xrange(len(nodes)):
        parent_id = rdw.sections[i + start_node][SEC.PID] - start_node
        if parent_id >= 0:
            nodes[parent_id].add_child(nodes[i])

    if post_action is not None:
        post_action(nodes[0])

    return nodes[0]


def load_neuron(filename, tree_action=set_neurite_type):
    '''Build section trees from an h5 file'''
    rdw = H5.read(filename, remove_duplicates=False, wrapper=SecDataWrapper)
    trees = [make_tree(rdw, trunk, tree_action)
             for trunk in rdw.neurite_trunks()]
    soma = make_soma(rdw.soma_points())
    return Neuron(soma, trees, rdw)


def soma_surface_area(nrn):
    '''Get the surface area of a neuron's soma.

    Note:
        The surface area is calculated by assuming the soma is spherical.
    '''
    return 4 * math.pi * nrn.soma.radius ** 2


def n_segments(nrn, neurite_type=NeuriteType.all):
    '''Number of segments in a section'''
    return sum(len(s.value) - 1
               for s in i_chain2(nrn.neurites, tree_filter=is_type(neurite_type)))


def n_sections(nrn, neurite_type=NeuriteType.all):
    '''Number of sections in a neuron'''
    return sum(1 for _ in i_chain2(nrn.neurites, tree_filter=is_type(neurite_type)))


def n_neurites(nrn, neurite_type=NeuriteType.all):
    '''Number of neurites in a neuron'''
    is_ntype = is_type(neurite_type)
    return sum(1 for n in nrn.neurites if is_ntype(n))


def get_section_lengths(nrn, neurite_type=NeuriteType.all):
    '''section lengths'''
    return [mm.path_distance(s.value)
            for s in i_chain2(nrn.neurites, tree_filter=is_type(neurite_type))]


def get_path_lengths(nrn, neurite_type=NeuriteType.all):
    '''Less naive path length calculation

    Calculates and stores the section lengths in one pass,
    then queries the lengths in the path length iterations.
    This avoids repeatedly calculating the lengths of the
    same sections.
    '''
    dist = {}

    for s in i_chain2(nrn.neurites, tree_filter=is_type(neurite_type)):
        dist[s] = mm.path_distance(s.value)

    def pl2(sec):
        '''Calculate the path length using cahced section lengths'''
        return sum(dist[s] for s in iupstream(sec))

    return [pl2(s) for s in i_chain2(nrn.neurites, tree_filter=is_type(neurite_type))]


def get_local_bifurcation_angles(nrn, neurite_type=NeuriteType.all):
    '''Get a list of all local bifurcation angles'''
    return [local_bifurcation_angle(b)
            for b in i_chain2(nrn.neurites,
                              iterator_type=ibifurcation_point,
                              tree_filter=is_type(neurite_type))]


def get_remote_bifurcation_angles(nrn, neurite_type=NeuriteType.all):
    '''Get a list of all remote bifurcation angles'''
    return [remote_bifurcation_angle(b)
            for b in i_chain2(nrn.neurites,
                              iterator_type=ibifurcation_point,
                              tree_filter=is_type(neurite_type))]


def get_section_radial_distances(nrn, neurite_type=NeuriteType.all, origin=None):
    '''Get a list of all remote bifurcation angles'''
    tree_filter = is_type(neurite_type)
    dist = []
    for n in nrn.neurites:
        if tree_filter(n):
            origin = n.value[0] if origin is None else origin
            dist.extend([section_radial_distance(s, origin) for s in ipreorder(n)])

    return dist


def get_trunk_section_lengths(nrn, neurite_type=NeuriteType.all):
    '''Get a list of the lengths of trunk sections of neurites in a neuron'''
    tree_filter = is_type(neurite_type)
    return [mm.path_distance(s.value) for s in nrn.neurites if tree_filter(s)]


def get_trunk_origin_radii(nrn, neurite_type=NeuriteType.all):
    '''Get a list of the lengths of trunk sections of neurites in a neuron'''
    tree_filter = is_type(neurite_type)
    return [s.value[0][COLS.R] for s in nrn.neurites if tree_filter(s)]


def get_n_sections_per_neurite(nrn, neurite_type=NeuriteType.all):
    '''Get the number of sections per neurite in a neuron'''
    tree_filter = is_type(neurite_type)
    return [sum(1 for _ in ipreorder(n)) for n in nrn.neurites if tree_filter(n)]


def section_path_length(section):
    '''Path length from section to root'''
    return sum(mm.path_distance(s.value) for s in iupstream(section))


def section_radial_distance(section, origin):
    '''Return the radial distances of a tree section to a given origin point

    The radial distance is the euclidian distance between the
    end-point point of the section and the origin point in question.

    Parameters:
        section: neurite section object
        origin: point to which distances are measured. It must have at least 3\
            components. The first 3 components are (x, y, z).
    '''
    return mm.point_dist(section.value[-1], origin)


def local_bifurcation_angle(bif_point):
    '''Return the opening angle between two out-going sections
    in a bifurcation point

    The bifurcation angle is defined as the angle between the first non-zero
    length segments of a bifurcation point.
    '''
    def skip_0_length(sec):
        '''Return the first point with non-zero distance to first point'''
        p0 = sec[0]
        cur = sec[1]
        for i, p in enumerate(sec[1:]):
            if not np.all(p[:COLS.R] == p0[:COLS.R]):
                cur = sec[i + 1]
                break

        return cur

    ch = (skip_0_length(bif_point.children[0].value),
          skip_0_length(bif_point.children[1].value))

    return mm.angle_3points(bif_point.value[-1], ch[0], ch[1])


def remote_bifurcation_angle(bif_point):
    '''Return the opening angle between two out-going sections
    in a bifurcation point

    The angle is defined as between the bofircation point and the
    last points in the out-going sections.

    '''
    return mm.angle_3points(bif_point.value[-1],
                            bif_point.children[0].value[-1],
                            bif_point.children[1].value[-1])
