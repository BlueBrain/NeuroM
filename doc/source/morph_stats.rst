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

neurom stats: morphometric statistics extraction
************************************************

The ``neurom stats`` application extracts morphometrics from a set of morphology
files and produces a summary in JSON or CSV format. It may obtain any of the morphometrics available
in the :py:func:`neurom.get` function, and is highly configurable, allowing the user to get
raw or summary statistics from a large set of neurite and morphology features.

An example usage:

.. code-block:: bash

    neurom stats path/to/morph/file_or_dir --config path/to/config --output path/to/output/file

For more information on the application and available options, invoke it with the ``--help``
or ``-h`` option.

.. code-block:: bash

    neurom stats --help

The functionality can be best explained by looking at a sample configuration file that is supposed
to go under ``--config`` option:

Config
------

Short format (prior version 3.0.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An example config:

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
        soma_radius:
            - mean


Here, there are two feature categories,

1. ``neurite``: these are morphometrics obtained from neurites, e.g. branch orders, section
   lengths, bifurcation angles, path lengths.
2. ``neuron``: these are morphometrics that can be applied to a whole morphology, e.g. the soma radius,
   the trunk radii, etc.

Each category sub-item (section_lengths, soma_radius, etc) corresponds to a :py:func:`neurom.get` feature, and each one of its sub-items corresponds to a statistic aggregating
function, e.g.

* ``raw``: array of raw values
* ``max``, ``min``, ``mean``, ``median``, ``std``: self-explanatory.
* ``total``: sum of the raw values

An additional field ``neurite_type`` specifies the neurite types into which the morphometrics
are to be split. It applies only to ``neurite`` features. A sample output using the above
configuration:

.. code-block:: json

    {
      "some/path/morph.swc":{
        "mean_soma_radius":0.17071067811865476,
        "axon":{
          "sum_section_lengths":207.87975220908129,
          "max_section_lengths":11.018460736176685,
          "max_section_branch_orders":10,
          "sum_section_volumes":276.73857657289523
        },
        "all":{
          "sum_section_lengths":840.68521442251949,
          "max_section_lengths":11.758281556059444,
          "max_section_branch_orders":10,
          "sum_section_volumes":1104.9077419665782
        },
        "apical_dendrite":{
          "sum_section_lengths":214.37304577550353,
          "max_section_lengths":11.758281556059444,
          "max_section_branch_orders":10,
          "sum_section_volumes":271.9412385728449
        },
        "basal_dendrite":{
          "sum_section_lengths":418.43241643793476,
          "max_section_lengths":11.652508126101711,
          "max_section_branch_orders":10,
          "sum_section_volumes":556.22792682083821
        }
      }
    }

.. _morph-stats-new-config:

Kwargs format (starting version 3.0.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The new format:

- requires to use ``morphology`` instead of ``neuron`` key in the config.
- requires to use ``sum`` instead of ``total`` statistic aggregating function.
- allows to specify features arguments.

For example, ``partition_asymmetry`` feature has additional arguments like ``method`` and ``variant``.
Before it wasn't possible to set them.
Here is how you can set them now:

.. code-block:: yaml

    neurite:
        partition_asymmetry:
            kwargs:
               variant: 'length'
               method: 'petilla'
            modes:
               - max
               - sum

Instead of statistic aggregating functions right after a feature name, config expects ``kwargs``
and ``modes`` properties. The former sets the feature arguments. The latter sets the statistic
aggregating function. This allows to set ``neurite_type`` directly on the feature, and overwrites
global setting of neurite types via ``neurite_type`` global config field. For example:

.. code-block:: yaml

    neurite:
        section_lengths:
            kwargs:
               neurite_type: APICAL_DENDRITE
            modes:
               - max
               - sum

So the example config from `Short format (prior version 3.0.0)`_ looks:

.. code-block:: yaml

    neurite:
        section_lengths:
            modes:
               - max
               - sum
        section_volumes:
            modes:
               - sum
        section_branch_orders:
            modes:
               - max

    neurite_type:
        - AXON
        - APICAL_DENDRITE
        - BASAL_DENDRITE
        - ALL

    morphology:
        soma_radius:
            modes:
               - mean


List of features format (starting version 3.2.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kwargs entry was converted into a list to allow running the same feature with different arguments. The ``partition_asymmetry`` feature in the example above can be specified multiple times with different arguments as follows:

.. code-block:: yaml

    neurite:
      partition_asymmetry:
        kwargs:
        - method: petilla
          variant: length
        - method: uylings
          variant: branch-order
        modes:
        - max
        - sum

To allow differentiation between the feature multiples, the keys and values of the kwargs are appended at the end of the feature name:

.. code-block::

    partition_asymmetry__variant:length__method:petilla
    partition_asymmetry__variant:branch-order__method:uylings

The example config from `Short format (prior version 3.0.0)`_ becomes:

.. code-block:: yaml

    neurite:
      section_branch_orders:
        kwargs:
        - {}
        modes:
        - max
      section_lengths:
        kwargs:
        - {}
        modes:
        - max
        - sum
      section_volumes:
        kwargs:
        - {}
        modes:
        - sum
    morphology:
      soma_radius:
        kwargs:
        - {}
        modes:
        - mean
    neurite_type:
    - AXON
    - APICAL_DENDRITE
    - BASAL_DENDRITE
    - ALL


Features
--------

All available features for ``--config`` are documented in :mod:`neurom.features.morphology`,
:mod:`neurom.features.neurite`, :mod:`neurom.features.population`.
