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

Morphology file data consistency checks
***************************************

.. code-block:: bash

    (nrm)$ morph_check some/data/path/morph_file.swc # single file
    INFO: ================================
    INFO: Check file some/data/path/morph_file.swc...
    INFO: Has valid soma? PASS
    INFO: Has Apical Dendrite? PASS
    INFO: Has Basal Dendrite? PASS
    INFO: All neurites have non-zero radius? PASS
    INFO: All segments have non-zero length? PASS
    INFO: All sections have non-zero length? PASS
    INFO: Check result: PASS
    INFO: ================================


    (nrm)$ morph_check some/data/path # all files in directory
    ....


Basic ``eyz.Neuron`` usage
**************************

- Load a neuron and obtain some information from it:

.. code-block:: python

    >>> from neurom import ezy
    >>> nrn = ezy.load_neuron('some/data/path/morph_file.swc')
    >>> apical_seg_lengths = nrn.get_segment_lengths(ezy.TreeType.apical_dendrite)
    >>> axon_sec_lengths = nrn.get_section_lengths(ezy.TreeType.axon)


- Visualize a neuronal morphology:

.. code-block:: python

    >>> # Initialize nrn as above
    >>> fig, ax = ezy.view(nrn)
    >>> fig.show()


Basic ``ezy`` examples script
*****************************

These basic examples illustrate the type of morphometrics that can be easily obtained
from the ``ezy`` module, without the need for any other ``neurom`` modules or tools.

The idea here is to pre-package the most common analyses so that users can obtain the
morphometrics with a very minimal knowledge of ``python`` and ``neurom``.

.. literalinclude:: ../../examples/ezy.py
    :lines: 30-


Advanced ``ezy`` examples script
********************************

These slightly more complex examples illustrate what can be done with the ``ezy``
module in combination with various generic iterators and simple morphometric functions.

The idea here is that there is a great deal of flexibility to build new analyses based
on some limited number of orthogonal iterator and morphometric components that can
be combined in many ways. Users with some knowledge of ``python`` and ``neurom`` can easily
implement code to obtain new morphometrics.

All of the examples in the previous sections can be implemented
in a similar way to those presented here.


.. literalinclude:: ../../examples/ezy_advanced.py
    :lines: 30-
