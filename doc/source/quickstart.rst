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

Quickstart
**********

Install
=======

Install the latest release:

.. code-block:: bash

    $ pip install neurom

Install a specific version:

.. code-block:: bash

    $ pip install neurom==1.2.3

.. note::
    It is recommended that you install ``NeuroM`` into a virtualenv.
    See :ref:`virtualenv setup<venv-label>` for details on how to set that up.

.. seealso::
    The :ref:`installation instructions<installation-label>` for more details
    and alternative installation methods.

Analyze, visualize, and check
=============================

The :py:mod:`neurom` module has various helper functions and command line applications
to simplify loading neuron morphologies from files into ``neurom`` data structures and
obtaining morphometrics, either from single or multiple neurons.
The functionality described here is limited, but it is hoped
that it will suffice for most analyses.

Extract morphometrics with :py:func:`neurom.get`
------------------------------------------------

These are some of the properties can be obtained for a single neurite type or for all
neurites regardless of type via the :py:func:`neurom.get` function:

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

The usage is simple:

.. code::

    import neurom as nm
    nrn = nm.load_neuron('some/data/path/morph_file0.swc')
    nrn_ap_seg_len = nm.get('segment_lengths', nrn, neurite_type=nm.APICAL_DENDRITE)
    pop = nm.load_neurons('some/data/path')
    pop_ap_seg_len = nm.get('segment_lengths', pop, neurite_type=nm.APICAL_DENDRITE)

This function also allows obtaining the soma radius and surface area.


Iterate over neurites with :py:func:`neurom.iter_neurites`
----------------------------------------------------------

The :py:func:`neurom.iter_neurites` function allows to iterate over the neurites
of a single neuron or a neuron population. It can also be applied to a single
neurite or a list of neurites. It allows to optionally pass a function to be
mapped onto each neurite, as well as a neurite filter function. In this example,
we apply a simple user defined function to the apical dendrites in a population:

.. code::

    import neurom as nm

    def user_func(neurite):
        print 'Analysinz neurite', neurite
        return len(neurite.points)

    stuff = [x for x in nm.iter_neurites(pop, user_func, lambda n : n.type == nm.APICAL_DENDRITE)]

.. seealso::
    The :py:mod:`neurom` documentation for more details and examples.

View neurons with :py:mod:`neurom.viewer`
-----------------------------------------

There are also helper functions to  plot a neuron in 2 and 3 dimensions.

The :py:func:`neurom.viewer.draw` function allows the user to make two and three-dimensional
plots of neurites, somata and neurons. It also has a dendrogram neuron plotting mode.

.. seealso::
    The :py:mod:`neurom.viewer` documentation for more details and examples.


Extract morphometrics into JSON files
-------------------------------------

The :doc:`morph_stats<morph_stats>` application lets you obtain various morphometrics
quantities from a set of morphology files. It is highly configurable, and gives access
to all the features available via the :py:func:`neurom.get` function.

For example,

.. code-block:: bash

    $ neurom stats some/path/morph.swc # single file
    {
      "some/path/morph.swc":{
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
        "mean_soma_radius":0.17071067811865476,
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

    $ neurom stats some/path # all files in directory

.. seealso::
    The :doc:`morph_stats documentation page<morph_stats>`


Check data validity
-------------------

The :doc:`morph_check<morph_check>` application applies some semantic
checks to morphology data files in order to
determine whether it is suitable to construct a neuron structure and whether certain
defects within the structure are detected. It can be invoked from the command line, and
takes as main argument the path to either a single file or a directory of morphology files.

For example,

.. code-block:: bash

    $ neurom check some/path/morph.swc # single file
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

    $ neurom check test_data/swc # all files in directory
    # loops over all morphology files found in test_data/swc

.. seealso::
    The :doc:`neurom check documentation page<morph_check>`
