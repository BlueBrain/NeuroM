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

morph_stats: morphometric statistics extraction
***********************************************

The ``neurom stats`` application extracts morphometrics from a set of neuron morphology
files and produces a summary in JSON or CSV format. It may obtain any of the morphometrics available
in the :py:func:`neurom.get` function, and is highly configurable, allowing the user to get
raw or summary statistics from a large set of neurite and neuron features.

An example usage:

.. code-block:: bash

    neurom stats path/to/morph/file_or_dir --config path/to/config --output path/to/output/file

The functionality can be best explained by looking at a sample configuration file that is supposed
to go under ``--config`` option:

.. code-block:: yaml
    
    neurite:
        section_lengths:
            - max
            - total
        section_volumes:
            - total
        section_branch_orders:
            - max
    
    neurite_type:
        - AXON
        - APICAL_DENDRITE
        - BASAL_DENDRITE
        - ALL
    
    neuron:
        soma_radii:
            - mean


Here, there are two feature categories,

1. ``neurite``: these are morphometrics obtained from neurites, e.g. branch orders, section
   lengths, bifurcation angles, path lengths.
2. ``neuron``: these are morphometrics that can be applied to a whole neuron, e.g. the soma radius,
   the trunk radii, etc.

Each category sub-item (section_lengths, soma_radii, etc) corresponds to a
:py:func:`neurom.get` feature, and each one of its sub-items corresponds to a statistic, e.g.

* ``raw``: array of raw values
* ``max``, ``min``, ``mean``, ``median``, ``std``: self-explanatory.
* ``total``: sum of the raw values
  
An additional field ``neurite_type`` specifies the neurite types into which the morphometrics
are to be split. This is a sample output using the above configuration:

.. code-block:: json

    {
      "some/path/morph.swc":{
        "mean_soma_radius":0.17071067811865476,
        "axon":{
          "total_section_length":207.87975220908129,
          "max_section_length":11.018460736176685,
          "max_section_branch_order":10,
          "total_section_volume":276.73857657289523
        },
        "all":{
          "total_section_length":840.68521442251949,
          "max_section_length":11.758281556059444,
          "max_section_branch_order":10,
          "total_section_volume":1104.9077419665782
        },
        "apical_dendrite":{
          "total_section_length":214.37304577550353,
          "max_section_length":11.758281556059444,
          "max_section_branch_order":10,
          "total_section_volume":271.9412385728449
        },
        "basal_dendrite":{
          "total_section_length":418.43241643793476,
          "max_section_length":11.652508126101711,
          "max_section_branch_order":10,
          "total_section_volume":556.22792682083821
        }
      }
    }


For more information on the application and available options, invoke it with the ``--help``
or ``-h`` option.

.. code-block:: bash

    neurom stats --help
    neurom --help  # to see all logging options

Features
--------

To see all available features for ``--config``:

.. runblock:: console

    $ neurom features
