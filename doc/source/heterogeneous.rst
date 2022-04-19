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

.. _heterogeneous:

Heterogeneous Morphologies
**************************

.. image:: images/heterogeneous_neuron.png

Definition
----------

A heterogeneous morphology consists of homogeneous and at least one heterogeneous neurite tree. A heterogeneous neurite tree consists of multiple sub-neurites with different types.

A typical example of a heterogeneous neurite is the axon-carrying dendrite, in which the axon sprouts from the basal dendrite.


Identification
--------------

Heterogeneous neurites can be identified using the ``Neurite.is_heterogeneous`` method:

.. code:: python

    from neurom import load_morphology
    from neurom.core.morphology import iter_neurites

    m = load_morphology('tests/data/swc/heterogeneous_morphology.swc')

    print([neurite.is_heterogeneous() for neurite in m])

which would return ``[False, True, False]`` in this case.


Sub-neurite views of heterogeneous neurites
--------------------------------------------

Default mode
~~~~~~~~~~~~

NeuroM does not take into account heterogeneous sub-neurites by default. A heterogeneous neurite is treated as a homogeneous one, the type of which is determined by the first section of
the tree. For example:

.. code-block:: python

    from neurom import load_morphology
    from neurom.core.morphology import iter_neurites

    m = load_morphology('tests/data/swc/heterogeneous_morphology.swc')

    basal, axon_carrying_dendrite, apical = list(iter_neurites(m))

    print(basal.type, axon_carrying_dendrite.type, apical.type)

would print the types ``(basal_dendrite, basal_dendrite, apical_dendrite)``, i.e. the axon-carrying dendrite would be treated as a basal dendrite. From feature extraction to checks, the axon-carrying dendrite is treated as a basal dendrite. Features, for which an axon neurite type is passed, do not have access to the axonal part of the neurite. For instance, the number of basal and axon neurites will be two and zero respectively.

Sub-neurite mode
~~~~~~~~~~~~~~~~

NeuroM provides an immutable approach (without modifying the morphology) to access the homogeneous sub-neurites of a neurite. Using ``iter_neurites`` with the flag ``use_subtrees`` activated returns a neurite view for each homogeneous sub-neurite.

.. code-block:: python

    basal1, basal2, axon, apical = list(iter_neurites(m, use_subtrees=True))

    print(basal1.type, basal2.type, axon.type, apical.type)

In the example above, two views of the axon-carrying dendrite have been created: the basal and axon dendrite views.

.. image:: images/heterogeneous_neurite.png

Given that the morphology is not modified, the sub-neurites specify as their ``root_node`` the section of the homogeneous sub-neurite. They are just pointers to where the sub-neurites start.

.. note::
    Creating neurite instances for the homogeneous sub-neurites breaks the assumption of root nodes not having a parent.


.. warning::
    Be careful while using sub-neurites. Because they just point to the start sections of the sub-neurite, they may include other sub-neurites as well. In the figure example above, the basal
    sub-neurite includes the entire tree, including the axon sub-neurite. An additional filtering of the sections is needed to leave out the axonal part. However, for the axon sub-neurite this
    filtering is not needed because it is downstream homogeneous.


Extract features from heterogeneous morphologies
------------------------------------------------

Neurite
~~~~~~~

Neurite features have been extended to include a ``section_type`` argument, which can be used to apply a feature on a heterogeneous neurite.

.. code-block:: python

    from neurom import NeuriteType
    from neurom import load_morphology
    from neurom.features.neurite import number_of_sections

    m = load_morphology('tests/data/swc/heterogeneous_morphology.swc')

    axon_carrying_dendrite = m.neurites[1]

    total_sections = number_of_sections(axon_carrying_dendrite)
    basal_sections = number_of_sections(axon_carrying_dendrite, section_type=NeuriteType.basal_dendrite)
    axon_sections = number_of_sections(axon_carrying_dendrite, section_type=NeuriteType.axon)

    print(total_sections, basal_sections, axon_sections)

Not specifying a ``section_type``, which is equivalent to passing ``NeuriteType.all``, will use all sections as done so far by NeuroM.

Morphology
~~~~~~~~~~

Morphology features have been extended to include the ``use_subtrees`` flag, which allows to use the sub-neurites.

.. code-block:: python

    from neurom import NeuriteType
    from neurom import load_morphology
    from neurom.features.morphology import number_of_neurites

    m = load_morphology('tests/data/swc/heterogeneous_morphology.swc')

    total_neurites_wout_subneurites = number_of_neurites(m)
    total_neurites_with_subneurites = number_of_neurites(m, use_subtrees=True)

    print(total_neurites_wout_subneurites, total_neurites_with_subneurites)

    number_of_axon_neurites_wout = number_of_neurites(m, neurite_type=NeuriteType.axon)
    number_of_axon_neurites_with = number_of_neurites(m, neurite_type=NeuriteType.axon, use_subtrees=True)

    print(number_of_axon_neurites_wout, number_of_axon_neurites_with)

    number_of_basal_neurites_wout = number_of_neurites(m, neurite_type=NeuriteType.basal_dendrite)
    number_of_basal_neurites_with = number_of_neurites(m, neurite_type=NeuriteType.basal_dendrite, use_subtrees=True)

    print(number_of_basal_neurites_wout, number_of_basal_neurites_with)

In the example above, the total number of neurites increases from 3 to 4 when the subtrees are enabled. This is because the axonal and basal parts of the axon-carrying dendrite are counted separately
in the second case.

Specifying a ``neurite_type``, allows to count sub-neurites. Therefore, the number of axons without subtrees is 0, whereas it is 1 when subtrees are enabled. However, for basal dendrites the number
does not change (2) because the axon-carrying dendrite is perceived as basal dendrite in the default case.

features.get
~~~~~~~~~~~~

``features.get`` can be used with respect to what has been mentioned above for neurite and morphology features.

.. code-block:: python

    from neurom import features
    from neurom import load_morphology

    m = load_morphology('tests/data/swc/heterogeneous_morphology.swc')

    features.get("number_of_neurites", m, use_subtrees=True)
    features.get("number_of_sections", m, section_type=NeuriteType.axon)

Conventions & Incompatibilities
-------------------------------

Heterogeneous Forks
~~~~~~~~~~~~~~~~~~~

A heterogeneous bifurcation/fork, i.e. a section with children of different types, is ignored when features on bifurcations are calculated. It is not meaningful to calculate features, such as bifurcation angles, on transitional forks where the downstream subtrees have different types.

Incompatible features with subtrees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following features are not compatible with subtrees:

* trunk_origin_azimuths
* trunk_origin_elevations
* trunk_angles

because they require the neurites to be rooted at the soma. This is not true for sub-neurites. Therefore, passing a ``use_subtrees`` flag, will result to an error.
