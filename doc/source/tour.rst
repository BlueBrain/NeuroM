:orphan:

A Tour of NeuroM (obsolete)
***************************

This is a more in-depth guide to ``NeuroM``. It will consist of explanations of the
overall design philosophy and the main components. The target audience consists of
users who want to perform analyses beyond what is described in the quick and easy
section.


Design philosophy
=================

NeuroM is designed to be lighweight, simple, testable and extensible. It consists
mainly of reusable components with limited functionality that can be combined to
construct more complex algorithms. It makes few assumptions about input data, and
prefers to fail early and loudly over implicitly fixing any data errors.

As a part of its design, it does a limited amount of processing on input data: it
attempts to re-arrange the data into a convenient internal format without adding
or removing any information. It can be roughly considered to consist of

#. A :ref:`data block<tour-data-layout>` containing re-arranged input data.
#. :ref:`Tree structures<tour-tree-label>` indexing sub-trees in the data block.
#. :ref:`Iterators<iterator-label>` providing different types of iteration over those 
   trees.
#. Functions to extract information from each iteration point.

It includes higher level classes representing the soma and a whole neuron. A neuron
is simply a collection of trees representing the neurites, and a soma.

.. _tour-data-layout:

Data Layout
===========

The internal data layout is simply an array of :ref:`points<point-label>`. It is
stored in memory as a 2-dimensional ``numpy`` array, with each row representing a
data point. This data block is constructed when reading in a morphology file in
any of the accepted formats. No information is added or removed from the input
data.

.. _tour-tree-label:

Trees
=====

Trees are hierachical, recursive structures used to represent individual
neurites in a morphology. NeuroM represents neurites as trees containing
measurement point information at each node. The role of a tree is
simply to maintain the hierarchy of its components. As will be shown, most of
the interesting work is done through a combination of different types of tree 
:ref:`iteration<iterator-label>`,
and different mapping and conversion functions.

.. _iterator-label:

Iterators
=========

NeuroM provides different means of iterating over a single tree. These provide
means to visit different nodes of the tree - e.g. all nodes, all nodes between a
given node and the root node, forking points, end points - but also different
groupings of nodes - e.g. sections (sequences of nodes between bifurcation
points), triplets (sequences of three consecutive nodes), segments (pairs of
consecutive nodes).

The idea behind this general approach is that most interesting features of a
tree can be obtained by mapping or reducing a suitable iteration sequence with a
suitable mapping or reducing function. The general form of a mapping is

.. code:: python

    tree = ....
    xs = [get_x(i) for i in some_iterator(tree)]

An example of a reduce operation is

.. code:: python

    tree = ....
    y = sum(get_y(i) for i in some_iterator(tree))

NeuroM provides various functions that can be applied to nodes, segments,
triplets sections (these take the place of get_x and get_y in the examples
above.) With knowledge of iterators and functions, it is possible to implement
great things in single lines of code. However, some helper functions have been
provided for the most common tree operations. We will get back to that later.
First, let's start by listing the different iterator types, before implementing
some real examples in the :ref:`cook-book section<cookbook-label>`.

Currently, these are the tree iterators provided in the
:py:mod:`neurom.core.tree` module.  They are functions that have a tree object
as parameter and return a suitable iterator:

* ipreorder: depth first pre-order traversal of nodes
* ipostorder: depth-first post-order traversal of nodes
* iupstream: iterate to root node of tree
* ileaf: leaf or end-nodes
* iforking_point: nodes with more than one child
* ibifurcation_point: nodes with two children
* isegment: pairs of consecutive nodes
* itriplet: triplets of consecutive nodes
* isection: sequences of points between forking points. These include the forking point. Points joining sections are repeated.

.. todo::
    Generate above list from docstrings

All of these iterators resolve to tree objects, but most analyses are interested in 
the data stored in each node of the tree. This is kept in a value field of the tree. 
To ease access to the data, and iterator adaptor is provided:

* val_iter

This transforms a tree iterator so that it converts trees to values. It works for nested 
structures, such as segments, triplets and sections. So for example, printing the radius 
of all leaves of a tree would be done like this:

.. code-block:: python

    from neurom.core.tree import ileaf, val_iter
    t = ... # a neurom.core.tree.Tree object
    for leaf in val_iter(ileaf(t)):
        print leaf[3] # radius is 4th component of data

.. _cookbook-label:

Cook-book
=========

Now, for some real life examples. These examples rely on trees. An easy way to get some
is to load a morphology file into a neuron object.

.. code:: python

    from neurom.io.utils import load_neuron
    nrn = load_neuron('test_data/swc/Neuron.swc')
    trees = nrn.neurites

We will assume ``trees`` has been obtained in a similar way in the following examples.

Get the total length of a tree
------------------------------

This can be achieved by summing the lengths of all the segments in the tree. For
this, we iterate over all segments, calculate each segment length, and sum all
lengths together:

.. code:: python

    from neurom.core.tree import isegment, val_iter
    from neurom.morpmath import segment_length
    tree = trees[0]
    tree_length = sum(segment_length(s) for s in val_iter(isegment(tree)))

Get the path length to an end-point
-----------------------------------

This is the distance between a leaf node and the root, and can be calculated by
iterating upstream from the leaf to the root, summing the distance as we go
along:

.. code:: python

    from neurom.core.tree import isegment, ileaf, iupstream, val_iter
    from neurom.morphmath import segment_length
    # for demonstration purposes, get the first leaf we find:
    tree = tree[0]
    first_leaf = next(ileaf(tree))
    # now iterate segment-wise, upstream, and sum the lengths
    path_len = sum(segment_length(s) for s in val_iter(isegment(first_leaf, iupstream)))


This example is conceptually the same as the previous one, except for one
crucial point: we start the iteration *from* a leaf node, and iterate *towards* the
root. This is the reason for the extra complexity:

* We use leaf iterator ileaf to get the first leaf node. This is somewhat
  beyond the scope of this example, but it is an interesting example of use of a
  different kind of iterator
* We iterate in segments using isegment, but we tell it
  to iterate upstream. That is what the second parameter to isegment does: it
  transforms the order of iteration.

A variant of the last example is to use the helper function
``neurom.core.tree.imap_val``. This is an iterator mapping function that transforms
the target of the iteration from a tree object to the data stored in the tree. In other
words, it applies ``val_iter`` internally:

.. code:: python

    from neurom.core.tree import isegment, ileaf, iupstream, imap_val
    from neurom.morphmath import segment_length

    first_leaf = ... # get a leaf of the tree (see previous example)

    path_len = sum(imap_val(segment_length, isegment(first_leaf, iupstream)))


If this all seems too complicated, remember that it is a general approach that
will allow you to do many more things other than getting the path length to the
root. But if that is all you care about, NeuroM has a packaged function for it:

.. code:: python

    from neurom.morphtree import path_length
    ...
    # assume leaf is a leaf node obtained by means that are irrelevant to this example
    path_len = path_length(leaf)
