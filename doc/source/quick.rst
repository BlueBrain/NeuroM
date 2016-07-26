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

The :py:mod:`neurom.fst` module
-------------------------------

The :py:mod:`neurom.fst` brings together various neurom components and helper functions
to simplify loading neuron morphologies from files into ``neurom`` data structures and
obtaining morphometrics, either from single or multiple neurons.
The functionality is limited, but it is hoped that it will suffice for most analyses. 

These are some of the properties can be obtained for a single neurite type or for all
neurites regardless of type via the :py:func:`neurom.fst.get` function:

* Segment lengths
* Section lengths
* Segment radii
* Number of sections
* Number of sections per neurite
* Number of neurites
* Number of segments
* Local and remote bifurcation angles
* Section path distances
* Section radial distances
* Section branch orders
* Total neurite length

This function also allows obtaining the soma radius and surface area.

There are also helper functions to  plot a neuron in 2 and 3 dimensions.

.. seealso::
    The :py:mod:`neurom.fst` documentation for more details and examples.

The :py:mod:`neurom.viewer` module
----------------------------------

The :py:func:`neurom.viewer.draw` function allows the user to make two and three-dimensional
plots of neurites, somata and neurons. It also has a dendrogram neurom plotting mode.

.. seealso::
    The :py:mod:`neurom.viewer` documentation for more details and examples.

Data checking application
-------------------------

The ``morph_check`` application applies some structural and semantic 
checks to morphology data files in order to
determine whether it is suitable to construct a neuron structure and whether certain
defects within the structure are detected. It can be invoked from the command line, and
takes as main argument the path to either a single file or a directory of morphology files.

For example,

.. code-block:: bash

    $ morph_check test_data/swc/Neuron.swc # single file
    INFO: ========================================
    INFO: File: test_data/swc/Neuron.swc
    INFO:                      Is single tree PASS
    INFO:                     Has soma points PASS
    INFO:                  No missing parents PASS
    INFO:                  Has sequential ids PASS
    INFO:                  Has increasing ids PASS
    INFO:                      Has valid soma PASS
    INFO:                  Has valid neurites PASS
    INFO:                  Has basal dendrite PASS
    INFO:                            Has axon PASS
    INFO:                 Has apical dendrite PASS
    INFO:     Has all nonzero segment lengths PASS
    INFO:     Has all nonzero section lengths PASS
    INFO:       Has all nonzero neurite radii PASS
    INFO:             Has nonzero soma radius PASS
    INFO:                                 ALL PASS
    INFO: ========================================

    $ morph_check test_data/swc # all files in directory
    # loops over all morphology files found in test_data/swc

The application also produces a summary json file, which can be useful when
processing more than one file:

.. code-block:: javascript

    {
        "files": {
            "test_data/swc/Neuron.swc": {
                "Is single tree": true,
                "Has soma points": true,
                "No missing parents": true,
                "Has sequential ids": true,
                "Has increasing ids": true,
                "Has valid soma": true,
                "Has valid neurites": true,
                "Has basal dendrite": true,
                "Has axon": true,
                "Has apical dendrite": true,
                "Has all nonzero segment lengths": true,
                "Has all nonzero section lengths": true,
                "Has all nonzero neurite radii": true,
                "Has nonzero soma radius": true,
                "ALL": true
            }
        },
        "STATUS": "PASS"
    }


The tests run are in submodules of :py:mod:`neurom.check`, particularly :py:mod:`structural_checks<neurom.check.structural_checks>`, :py:mod:`neurite_checks<neurom.check.neurite_checks>` and
:py:mod:`soma_checks<neurom.check.soma_checks>`.


For more information, use the help option:

.. code-block:: bash

    $ morph_check --help
    ....
