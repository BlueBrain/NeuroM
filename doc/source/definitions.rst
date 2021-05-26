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


.. _definitions-label:

NeuroM morphology definitions
=============================

``NeuroM`` specific working definitions of various components of neuron morphologies.


.. _point-label:

Point
-----

A point is a ``numpy`` array of numbers **[X, Y, Z, R]** where the components are

* X, Y, Z: Cartesian coordinates of position
* R: Radius

.. _segment-label:

Segment
-------

A segment consists of two consecutive :ref:`points<point-label>` belonging to
the same :ref:`neurite<neurite-label>` and :ref:`section<section-label>`.

In ``NeuroM`` a segment is represented as a tuple or a ``numpy`` array of two `points<point-label>`.


.. _section-label:

Section
-------

A section is a node of morphology tree containing a series of two or more :ref:`points<point-label>`
whose first and last element are any of the following combinations:

* root node, forking point
* forking point, forking point
* forking point, end point
* root node, end point

The first point of a section is a duplicate of the last point of its parent section,
unless the latter is a soma section.

In ``NeuroM``, a section is represented by class :class:`Section<neurom.core.neuron.Section>`.

.. _soma-label:

Soma
----

A soma can be represented by one or more :ref:`points<point-label>`.
The soma is classified solely based on the number of points it contains thus:

* Type A: 1 point defining the center and radius.
* Type B: 3 points. Only the centers of the points are considered.
  The first point defines the center. The radius is estimated from
  the mean distance between the center and the two remaining points.
* Type C: More than three points. The center is defined as the mean position
  of all points. The radius is defined as the mean distance of all points to
  the center.

.. todo::
    Expand list if and when specifications require new types of soma.

The soma is represented by classes derived from :py:class:`Soma<neurom.core.soma.Soma>`.
The interface exports a center and radius. These can be calculated in different
ways, but the default is to use the center and radius for type A and the mean center
and radius for types B and C.

.. todo::
    In the future, type B may be interpreted as 3 points on an ellipse.
    In this case, the points would have to be non-collinear.
    Currently there is no such restriction.

See also

.. seealso:: The :py:mod:`soma implementation module<neurom.core.soma>`


.. _neurite-label:

Neurite tree
------------

A neurite is essentially a tree of :ref:`sections<section-label>`. The tree structure
implies the following:

* A node can only have one parent.
* A node can have an arbitrary number of children.
* No loops are present in the structure.

Neurites are represented by the class :py:class:`Neurite<neurom.core.Neurite>`, which contains
the root node of the aforementioned tree as well as some helper functions to aid iteration
over sections and collection of points.

In :py:mod:`NeuroM<neurom>` neurite trees are implemented using the recursive structure
:py:class:`neurom.core.Section`, :ref:`described above<section-label>`.


Neuron
------

A neuron structure consists of a single :ref:`soma<soma-label>` and a collection of
:ref:`neurites<neurite-label>`.

The trees that are expected to be present depend on the type of cell:

* Interneuron (IN): basal dendrite, axon
* Pyramidal cell (PC): basal dendrite, apical dendrite, axon

Neurons are represented by the class :py:class:`Neuron<neurom.core.Neuron>`. This is more
or less what it looks like:

.. code-block:: python

    neuron = {
        soma,
        neurites,
        points,
        name
    }
