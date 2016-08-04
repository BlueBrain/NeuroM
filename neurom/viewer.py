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


'''Tools to visualize neuron morphological objects

Examples:

    >>> from neurom import viewer
    >>> nrn = ... # load a neuron
    >>> viewer.draw(nrn)                    # 2d plot
    >>> viewer.draw(nrn, mode='3d')         # 3d plot
    >>> viewer.draw(nrn.neurites[0])        # 2d plot of neurite tree
    >>> viewer.draw(nrn, mode='dendrogram') # dendrogram plot

'''

from .view.view import neuron as draw_neuron
from .view.view import neuron3d as draw_neuron3d
from .view.view import tree as draw_tree
from .view.view import tree3d as draw_tree3d
from .view.view import soma as draw_soma
from .view.view import soma3d as draw_soma3d
from .view.view import dendrogram as draw_dendrogram
from .core import Soma, Neuron
from .fst import Neurite, Tree


MODES = ('2d', '3d', 'dendrogram')

_VIEWERS = {
    'neuron_3d': draw_neuron3d,
    'neuron_2d': draw_neuron,
    'neuron_dendrogram': draw_dendrogram,
    'tree_3d': draw_tree3d,
    'tree_2d': draw_tree,
    'tree_dendrogram': draw_dendrogram,
    'soma_3d': draw_soma3d,
    'soma_2d': draw_soma
}


class ViewerError(Exception):
    '''Base class for viewer exceptions'''
    pass


class InvalidDrawModeError(ViewerError):
    '''Exception class to indicate invalid draw mode'''
    pass


class NotDrawableError(Exception):
    '''Exception class for things that aren't drawable'''
    pass


def draw(obj, mode='2d', **kwargs):
    '''Draw a morphology object

    Parameters:
        obj: morphology object to be drawn (neuron, tree, soma).
        mode (Optional[str]): drawing mode ('2d', '3d', 'dendrogram'). Defaults to '2d'.
        **kwargs: keyword arguments for underlying neurom.view.view functions.

    Raises:
        InvalidDrawModeError if mode is not valid
        NotDrawableError if obj is not drawable
        NotDrawableError if obj type and mode combination is not drawable

    Examples:

        >>> nrn = ... # load a neuron
        >>> fig, _ = viewer.draw(nrn)             # 2d plot
        >>> fig.show()
        >>> fig3d, _ = viewer.draw(nrn, mode='3d') # 3d plot
        >>> fig3d.show()
        >>> fig, _ = viewer.draw(nrn.neurites[0]) # 2d plot of neurite tree
        >>> dend, _ = viewer.draw(nrn, mode='dendrogram')

    '''

    if mode not in MODES:
        raise InvalidDrawModeError('Invalid drawing mode %s', mode)

    if isinstance(obj, Neuron):
        tag = 'neuron'
    elif isinstance(obj, (Tree, Neurite)):
        tag = 'tree'
    elif isinstance(obj, Soma):
        tag = 'soma'
    else:
        raise NotDrawableError('draw not implemented for %s', obj.__class__)

    viewer = '%s_%s' % (tag, mode)
    try:
        return _VIEWERS[viewer](obj, **kwargs)
    except KeyError:
        raise NotDrawableError('No drawer for class %s, mode=%s' % (obj.__class__, mode))
