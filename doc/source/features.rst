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

.. _features:

Features
********
A tool for analysing of morphologies. It allows to extract various information about morphologies.
For example if you need to know the segment lengths of a morphology then you need to call
``segment_lengths`` feature. The complete list of available features is spread among pages
:mod:`neurom.features.neurite`, :mod:`neurom.features.morphology`,
:mod:`neurom.features.population`.

Features are spread among ``neurite``, ``morphology``, ``population`` modules to emphasize their
expected input. Features from ``neurite`` expect a neurite as their input. So calling it with
a morphology input will fail. ``morphology`` expects a morphology only. ``population`` expects a
population only.

This restriction can be bypassed if you call a feature from ``neurite`` via the features
mechanism ``features.get``. However the mechanism does not allow to appply ``population``
features to anything other than a morphology population, and ``morphology`` features can be applied
only to a morphology or a morphology population.

An example for ``neurite``:

.. testcode::

    from neurom import load_morphology, features
    from neurom.features.neurite import max_radial_distance

    m = load_morphology("tests/data/swc/Neuron.swc")

    # valid input
    rd = max_radial_distance(m.neurites[0])

    # invalid input
    # rd = max_radial_distance(m)

    # valid input
    rd = features.get('max_radial_distance', m)


The features mechanism assumes that a neurite feature must be summed if it returns a number, and
concatenated if it returns a list. Other types of returns are invalid. For example lets take
a feature ``number_of_segments`` of ``neurite``. It returns a number of segments in a neurite.
Calling it on a morphology will return a sum of ``number_of_segments`` of all the morphology's neurites.
Calling it on a morphology population will return a list of ``number_of_segments`` of each morphology
within the population.


.. testcode::

   from neurom import load_morphology, load_morphologies, features

   m = load_morphology("tests/data/swc/Neuron.swc")

   # a single number
   features.get('number_of_segments', m.neurites[0])

   # a single number that is a sum for all `m.neurites`.
   features.get('number_of_segments', m)

   pop = load_morphologies("tests/data/valid_set")

   # a list of numbers
   features.get('number_of_segments', pop)

if a list is returned then the feature results are concatenated.

.. testcode::

   from neurom import load_morphology, load_morphologies, features

   m = load_morphology("tests/data/swc/Neuron.swc")

   # a list of lengths in a neurite
   features.get('section_lengths', m.neurites[0])

   # a flat list of lengths in a morphology, no separation among neurites
   features.get('section_lengths', m)

   pop = load_morphologies("tests/data/valid_set")

   # a flat list of lengths in a population, no separation among morphologies
   features.get('section_lengths', pop)

In case such implicit behaviour does not work a feature can be rewritten for each input separately.
For example, a feature ``max_radial_distance`` that requires a `max` operation instead of implicit
`sum`. Its definition in ``neurite``:

.. literalinclude:: ../../neurom/features/neurite.py
    :pyobject: max_radial_distance

In order to make it work for a morphology, it is redefined in ``morphology``:

.. literalinclude:: ../../neurom/features/morphology.py
    :pyobject: max_radial_distance

Another feature that requires redefining is ``sholl_frequency``. This feature applies different
logic for a morphology and a morphology population. That is why it is defined in ``morphology``:

.. literalinclude:: ../../neurom/features/morphology.py
    :pyobject: sholl_frequency

and redefined in ``population``

.. literalinclude:: ../../neurom/features/population.py
    :pyobject: sholl_frequency
