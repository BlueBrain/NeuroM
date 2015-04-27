NeuroM
======

TODO: write package desctiption and detailed installation instructions.


Installation
============

NeuroM is a ``pip``-installable package, so it is possible to install it system-wide or
into a ``virtualenv`` using ``pip``:

    $ pip install hbp-neurom-X.Y.Z.tgz

This assumes the dependencies have already been installed.


Dependencies
============

NeuroM depends on numpy and matplotlib. These packages use natively compiled libraries,
so it is best if they are installed system-wide, taking advantage of system optimized
numeric libraries such as LAPACK and BLAS. However, both are available as
``pip``-installable packages, so in principle can be built into a ``virtualenv``. Note
that the build time may be considerable.


Installation and runtime
------------------------

* ``numpy >= 1.8.0``
* ``matplotlib >= 1.3.1``

Testing
-------

* ``nose``
* ``coverage``
