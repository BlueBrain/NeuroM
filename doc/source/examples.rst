.. Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
   All rights reserved.

   This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

       1. Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
       2. Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
       3. Neither the name of the copyright holder nor the names of
          its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Examples
========

Fast analysis with :py:mod:`neurom`
***********************************

Here we load a neuron and obtain some information from it:

.. code-block:: python

    >>> import neurom as nm
    >>> nrn = nm.load_neuron('some/data/path/morph_file.swc')
    >>> ap_seg_len = nm.get('segment_lengths', nrn, neurite_type=nm.APICAL_DENDRITE)
    >>> ax_sec_len = nm.get('section_lengths', nrn, neurite_type=nm.AXON)


Morphology visualization with the :py:mod:`neurom.viewer` module
****************************************************************

Here we visualize a neuronal morphology:


.. code-block:: python

    >>> # Initialize nrn as above
    >>> from neurom import viewer
    >>> fig, ax = viewer.draw(nrn)
    >>> fig.show()
    >>> fig, ax = viewer.draw(nrn, mode='3d') # valid modes '2d', '3d', 'dendrogram'
    >>> fig.show()

Basic feature extraction example
********************************

These basic examples illustrate the type of morphometrics that can be easily obtained
directly from the ``neurom`` module, without the need for any other ``neurom``
sub-modules or tools.

The idea here is to pre-package the most common analyses so that users can obtain the
morphometrics with a very minimal knowledge of ``python`` and ``neurom``.

.. literalinclude:: ../../examples/get_features.py
    :lines: 30-


Advanced iterator-based feature extraction example
**************************************************

These slightly more complex examples illustrate what can be done with the ``neurom``
module's various generic iterators and simple morphometric functions.

The idea here is that there is a great deal of flexibility to build new analyses based
on some limited number of orthogonal iterator and morphometric components that can
be combined in many ways. Users with some knowledge of ``python`` and ``neurom`` can easily
implement code to obtain new morphometrics.

All of the examples in the previous sections can be implemented
in a similar way to those presented here.


.. literalinclude:: ../../examples/neuron_iteration_analysis.py
    :lines: 30-
