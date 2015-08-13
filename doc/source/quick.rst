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

Quick and easy analysis
=======================



The :py:mod:`neurom.ezy` module
-------------------------------

The :py:mod:`neurom.ezy` module contains a :py:class:`Neuron<neurom.ezy.neuron.Neuron>` class that allows to easily
load neuron morphology from a file into ``neurom`` data structures. It provides convenient
methods to query various properties of the neuron. The functionality is limited, but it
is hoped that it will suffice for most analyses. 

These are some of the properties can be obtained for a single neurite type or for all
neurites regardless of type:

* Segment lengths
* Section lengths
* Number of sections
* Number of sections per neurite
* Number of neurites
* Local and remote bifurcation angles
* Section path distances
* Section radial distances

There are also methods for obtaining the soma radius and for plotting a neuron
in 2 and 3 dimensions.

See :py:class:`neurom.ezy.neuron.Neuron` for more details and examples.

Data checking applications
--------------------------

There are two user-friendly data checking applications. ``raw_data_check`` checks for basic 
consistency
of raw data, and ``morph_check`` applies some further semantic checks to the data in order to
determine whether it is suitable to construct a neuron structure and whether certain
defects within the structure are detected. Both can be invoked from the command line, and
take as main argument the path to either a single file or a directory of morphology files.

For example,

.. code-block:: bash

    $ morph_check test_data/swc/Neuron.swc # single file
    INFO: ================================
    INFO: Check file test_data/swc/Neuron.swc...
    INFO: Has valid soma? PASS
    INFO: Has Apical Dendrite? PASS
    INFO: Has Basal Dendrite? PASS
    INFO: All neurites have non-zero radius? PASS
    INFO: All segments have non-zero length? PASS
    INFO: All sections have non-zero length? PASS
    INFO: Check result: PASS
    INFO: ================================


    $ morph_check test_data/swc # all files in directory
    # loops over all morphology files found in test_data/swc

For more information, use the help option:

.. code-block:: bash

    $ morph_check --help
    ....

    $ raw_data_check --help
    ....

