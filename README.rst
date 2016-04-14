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

NeuroM
******

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies.


.. image:: https://travis-ci.org/BlueBrain/NeuroM.svg?branch=master
    :target: https://travis-ci.org/BlueBrain/NeuroM
    :alt: Test Status

.. image:: http://codecov.io/github/BlueBrain/NeuroM/coverage.svg
    :target: http://codecov.io/github/BlueBrain/NeuroM
    :alt: Test Coverage Status

.. image:: https://readthedocs.org/projects/neurom/badge/?version=latest
    :target: http://neurom.readthedocs.org/en/latest/
    :alt: Documentation Status

Documentation
=============

NeuroM documentation is built and hosted on `readthedocs <https://readthedocs.org/>`_.

* `latest snapshot <http://neurom.readthedocs.org/en/latest/>`_
* `latest release <http://neurom.readthedocs.org/en/stable/>`_

Dependencies
============

The build-time and runtime dependencies of NeuroM are:

* numpy
* scipy
* matplotlib
* h5py (optional, required for reading HDF5 files)
* enum34 (pip install takes care of this)
* pyyaml (pip install takes care of this)

Installation
============

It is recommended that you use `pip <https://pip.pypa.io/en/stable/>`_ to install into
``NeuroM`` into a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_:

NeuroM installation
-------------------

The following assumes ``virtualenv`` named ``nrm`` with access to the dependencies has been set up
and activated.
We will see two ways to install ``NeuroM``

#. From the Python Package Index
#. From the git repository
#. From source (for NeuroM developers)

Install package from Python Package Index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    (nrm)$ pip install neurom

Install package from git
^^^^^^^^^^^^^^^^^^^^^^^^

Install the latest version:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git

Install a particular release:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git@neurom-v0.0.1

Install from source
^^^^^^^^^^^^^^^^^^^

Clone the repository and install it:

.. code-block:: bash

    (nrm)$ git clone https://github.com/BlueBrain/NeuroM.git
    (nrm)$ pip install -e ./NeuroM

This installs ``NeuroM`` into your ``virtualenv`` in "editable" mode. That means changes you make to the source code are seen by the installation.
To install in read-only mode, omit the ``-e``.

Examples
========

- Perform checks on neuron morphology files:

.. code-block:: bash

    (nrm)$ morph_check some/data/path/morph_file.swc # single file
    INFO: ================================
    INFO: Check file some/data/path/morph_file.swc...
    INFO: Has valid soma? PASS
    INFO: Has Apical Dendrite? PASS
    INFO: Has Basal Dendrite? PASS
    INFO: All neurites have non-zero radius? PASS
    INFO: All segments have non-zero length? PASS
    INFO: All sections have non-zero length? PASS
    INFO: Check result: PASS
    INFO: ================================


    (nrm)$ morph_check some/data/path # all files in directory
    ....

- Load a neuron and obtain some information from it:

.. code-block:: python

    >>> from neurom import ezy
    >>> nrn = ezy.load_neuron('some/data/path/morph_file.swc')
    >>> apical_seg_lengths = ezy.get('segment_lengths', nrn, ezy.NeuriteType.apical_dendrite)
    >>> axon_sec_lengths = ezy.get('section_lengths', nrn, ezy.NeuriteType.axon)


- Visualize a neuronal morphology:

.. code-block:: python

    >>> # Initialize nrn as above
    >>> fig, ax = ezy.view(nrn)
    >>> fig.show()

