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

.. NeuroM spported format documentation

Supported file formats
======================

NeuroM currently supports the SWC format, the BBP HDF5 formats, and offers
experimental support for NeuroLucida .asc files.

.. seealso::
    :ref:`The morphology definitions page<definitions-label>` for definitions of
    concepts such as :ref:`point<point-label>`, :ref:`section<section-label>`,
    :ref:`soma<soma-label>` and :ref:`neurite<neurite-label>` in NeuroM.

.. todo::
    Complete this section with additional NeuroM specific restrictions
    on the formats below.

SWC
---

The SWC format represents a neuron as a tree of 3D points with additional information.
More information can be found `here <http://research.mssm.edu/cnic/swc.html>`_.

.. todo::
    Add reference to SWC paper and more semantic constraints.

.. todo::
    Add semantic constraints on different soma types once these have been
    determined. For more info on what is to be considered, see
    `neuromorpho.org's <http://neuromorpho.org/SomaFormat.html>`_.

HDF5
----

The HDF5 morphology formats developed by the BBP represent the neuron as a tree of
:ref:`sections<section-label>`. The specifications for the two versions of the format
cn be found in `the HBP morphology format documentation page <https://developer.humanbrainproject.eu/docs/projects/morphology-documentation/0.0.2/index.html>`_.

NeuroLucida (experimental)
--------------------------

The `NeuroLucida <http://www.mbfbioscience.com/neurolucida>`_ .asc file format is commonly
used but lacking in an open format specification. NeuroM provides a best-effort experimental
reader that parses information equivalent to the two formats above, that is to say, it does
not deal with annotations or other meta-data, and is restricted purely to the topological and
geometrical features of a neuron, as well as the neurite type information.

.. warning::
    The NeuroLucida parser is experimental. Use at own risk when extracting numerical
    information. We make no statement as to the correctness of numerical output.

.. todo::
    References and more information?
