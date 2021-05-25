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

.. _migration-v2:

=======================
Migration to v2 version
=======================

- ``Neuron`` object now extends ``morphio.Morphology``.
- NeuroM does not remove unifurcations on load. Unifurcation is a section with a single child. Such
  sections are possible in H5 and ASC formats. Now, in order to remove them on your morphology, you
  would need to call ``remove_unifurcations()`` right after the morphology is constructed.

  .. code-block:: python

      import neurom as nm
      nrn = nm.load_neuron('some/data/path/morph_file.asc')
      nrn.remove_unifurcations()

- Soma is not considered as a section anymore. Soma is skipped when iterating over neuron's
  sections. It means that section indexing offset needs to be adjusted by
  ``-(number of soma sections)`` which is usually ``-1``.
- drop ``benchmarks``
- drop ``neurom.check.structural_checks`` as MorphIO does not allow to load invalid morphologies,
  and it does not give access to raw data.
- drop ``Tree`` class. Use ``Section`` instead as it includes its functionality but if you need
  ``Tree`` separately then copy-paste ``Tree`` code from v1 version to your project.
- ``Section`` and ``Neurite`` class can't be copied anymore because their underlying MorphIO
  objects can't be copied (pickled). Only copying of ``Neuron`` is preserved.
- drop ``FstNeuron``. It functionality is included in ``Neuron`` class. Use ``Neuron`` instead of
  ``FstNeuron``.
- Validation of morphologies changed.
    The following is not an invalid morphology anymore:

    - 2 point soma
    - non-sequential ids
- script ``morph_check`` and ``morph_stats`` changed to ``neurom check`` and ``neurom stats``
    correspondingly.