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

morph_check: the morphology checker
***************************************

The ``morph_check`` application performs checks on reconstructed morphologies from
data contained in morphology files, and so may be used as a morphology validation
of sorts.

The tests are grouped in two categories:

1. Structural tests. These apply to the structure of the data and are a good indicator
   as to whether a neuron object or any of its sub-components can actually be reconstructed.
   Failure in some of these may make further tests fail.
2. Neuron tests. These are applied to properties of reconstructed neurons and their
   constituent soma and neurites, and can be thought of as "quality" checks.


It is very likely that a failure in the structural tests will make the neuron
tests fail. Furthermore, inability to build a soma typically results
in an inability to build neurites. Failure to build a soma or neurites results
in an early failure for a given morphology file.

The application may be invoked with a ``YAML`` configuration file specifying which
checks to perform. The structure of the configuration file reflects the test categories
mentioned above. Here is an example configuration:

.. code-block:: yaml

    checks:
        structural_checks:
            - is_single_tree
            - has_soma_points
            - has_valid_soma
            - has_valid_neurites
        neuron_checks:
            - has_basal_dendrite
            - has_axon
            - has_all_nonzero_segment_lengths
            - has_all_nonzero_section_lengths
            - has_all_nonzero_neurite_radii
            - has_nonzero_soma_radius

    options :
        has_nonzero_soma_radius         : 0.0
        has_all_nonzero_neurite_radii   : 0.007
        has_all_nonzero_segment_lengths : 0.01
        has_all_nonzero_section_lengths : 0.01


As can be seen, the configuration file is split into two sections ``checks``, and ``options``.
Each of the ``checks`` sub-items corresponds to a sub-module of :py:mod:`neurom.check`, namely
:py:mod:`structural_checks<neurom.check.structural_checks>`
and :py:mod:`neuron_checks<neurom.check.neuron_checks>`. And each
of their sub-items corresponds to a function in that sub-module. This illustrates the possible
checks that may be applied by ``morph_check``.


The application also produces a summary ``json`` file, which can be useful when
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

For more information on the application and available options, invoke it with the ``--help``
or ``-h`` option.


.. code-block:: bash

    morph_check --help
