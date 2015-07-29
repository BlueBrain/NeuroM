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

NeuroM morphology definitions
=============================

These are ``NeuroM`` specific working definitions of various components of
neuron morpholigies.

This is to be based on `Basic Definitions to be used in NeuroM <https://bbpteam.epfl.ch/project/spaces/display/BBPSUBSIM/Basic+Definitions+to+be+used+in+NeuroM>`_

.. _point-label:

Point
-----

A point is a vector of numbers **[X, Y, Z, R, TYPE, ID, PID]** where the components are

* X, Y, Z: Cartesian coordinates of position
* R: Radius
* TYPE: One of the :py:class:`NeuroM valid point types<neurom.core.dataformat.POINT_TYPE>`
* ID: Unique identifier of the point.
* PID: ID of the parent of the point.

Typically only the first four or five components are of interest to morphology analysis.
The rest are used to construct the soma and hierarchical tree structures of the neuron,
and to check its semantic validity. 

.. note::
    For most of what follows, it suffices to consider a
    **point** as a vector of **[X, Y, Z, R, TYPE]**. The remaining
    components **ID** and **PID** can be considered book-keeping.

.. todo::
    Point types may need to be restricted to align SWC with H5. This is dependent on
    future H5 specs.


.. _soma-label:

Soma
----

A soma can be represented by one, three or more :ref:`points<point-label>`. 
The soma is classified based on
the number of points it contains thus:

* Type A: 1 point defining the center and radius.
* Type B: 3 points. Only the centers of the points are considered.
  The first point defines the center. The radius is extimated from
  the mean distance between the center and the two remaining points.
* Type C: More than three points. Only the centers of the points are considered.
  The first point defines the center. The radius is
  estimated from the mean distance between the center and the remaining points.

.. todo::
    Expand list if and when specifications require new types of soma.

The soma interface exports a center and radius. These can be calculated in different
ways, but the default is to use the center and radius for type A and the mean center
and radius for types B and C.

.. todo::
    In the future, type B may be interpreted as 3 points on an ellipse.
    In this case, the points would have to be non-collinear.
    Currently there is no such restriction.

See also

.. seealso:: :py:class:`neurom.core.neuron.SOMA_TYPE`


.. _tree-label:

Neurite tree
------------

A neurite consists of a tree structure with a set of :ref:`points<point-label>` in each
vertex or node. The tree structure implies the following:

* A node can only have one parent.
* A node can have an arbitrary number of children.
* No loops are present in the structure.

.. todo::
    Should neurite trees be restricted to being binary trees? If so, no more
    than two children per node would be allowed.

Different type of points are allowed in the same tree as long as same conventions
are followed

.. todo::
    The conventions governing the types of points in a neurite
    tree need to be well defined

In ``NeuroM`` neurite trees are implemented using the recursive structure 
:py:class:`neurom.core.tree.Tree`, with each node holding a reference to a
:ref:`morphology point<point-label>`.

Neuron
------

A neuron structure consists of a single :ref:`soma<soma-label>` and a collection of 
:ref:`trees<tree-label>`.

The trees that are expected to be present depend on the type of cell:

* Interneuron (IN): basal dendrite, axon
* Pyramidal cell (PC): basal dendrite, apical dendrite, axon

.. seealso::
    :py:class:`neurom.core.neuron.Neuron`
    :py:class:`neurom.ezy.Neuron`
