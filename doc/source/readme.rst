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

.. image:: https://travis-ci.org/BlueBrain/NeuroM.svg?branch=master
    :target: https://travis-ci.org/BlueBrain/NeuroM

Introduction
============

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies.

This quick start guide contains information on how to install NeuroM and use-case examples.

Dependencies
============

The build-time and runtime dependencies of NeuroM are:

* numpy
* h5py
* scipy
* matplotlib
* enum34

Additional dependencies needed for testing and building documentation are:

* nose
* coverage
* sphinx
* GNU Make

Installation
============

It is recommended that you use `pip <https://pip.pypa.io/en/stable/>`_ to install into
``NeuroM`` into a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_:

Virtualenv setup
----------------

.. code-block:: bash

    $ virtualenv --system-site-packages nrm   # creates a virtualenv called "nrm" in nrm directory
    $ source nrm/bin/activate                 # activates virtualenv
    (nrm)$                                    # now we are in the nrm virtualenv

Here, the ``--system-site-packages`` option has been used. This is because dependencies such as
``matplotlib`` aren't trivial to build in a ``virtualenv``. This setting allows python packages
installed in the system to be used inside the ``virtualenv``.

The prompt indicates that the ``virtualenv`` has been activated. To de-activate it,

.. code-block:: bash

    (nrm)$ deactivate

Note you do not have to work in the ``nrm`` directory. This is where python packages will get installed, but you can work anywhere on your file system, as long as you have activated the ``virtualenv``.

.. note::

    In following code samples, the prompts ``(nrm)$`` and ``$`` are used to indicate
    that the user virtualenv is *activated* or *deactivated* respectively.

.. note::

    In following code samples, the prompt ``>>>`` indicates a python interpreter session
    started *with the virtualenv activated*. That gives access to the ``neurom``
    installation.

NeuroM installation
-------------------

Once the ``virtualenv`` is set up, there are three ways to install ``NeuroM``

#. From the internal BBP PyPI server (restricted access)
#. From the git repository
#. From source (for NeuroM developers)

Install from the BBP PyPI server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the latest release:

.. code-block:: bash

    (nrm)$ pip install -i http://bbpgb019.epfl.ch:9090/simple neurom

Install a specific version:

.. code-block:: bash

    (nrm)$ pip install -i http://bbpgb019.epfl.ch:9090/simple neurom==1.2.3

.. warning::

    If your version of pip 7.0 or higher, you need to add the option
    ``--trusted-host bbpgb019.epfl.ch``.

Install from git
^^^^^^^^^^^^^^^^

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

Running the tests
-----------------

The tests require that you have cloned the repository, since the test code is
not distributed in the package. It is recommended to use ``nosetests`` for
this. There are two options:

Use the provided ``Makefile`` to run the tests using ``make``:

.. code-block:: bash

    $ git clone https://github.com/BlueBrain/NeuroM.git
    $ cd NeuroM
    $ make test

This runs ``pep8``, ``pylint`` and the unit tests in sequence.

The ``Makefile`` also has targets for running only pylint and pep8 individually:

.. code-block:: bash

        $ make lint       # runs pep8 and pylint if that succeeds
        $ make run_pep8   # run only pep8
        $ make run_pylint # run only pep8

This creates its own virtualenv ``neurom_test_venv`` and runs all the tests inside of
it.

Alternatively, inside the your own virtualenv, install ``nose`` and ``coverage``
if you haven't
done so already or these aren't installed in the system:

.. code-block:: bash

    (nrm)$ pip install nose
    (nrm)$ pip install coverage
    (nrm)$ nosetests -s -v --with-coverage --cover-package neurom

Building the Documentation
--------------------------

The documentation requires that you clone the repository. Once you have done that,
there's a ``make`` target to build the HTML version of the documentation:

.. code-block:: bash

    $ cd NeuroM # repository location
    $ make doc

This builds the documentation in ``doc/build``.
To view it, point a browser at ``doc/build/html/index.html``

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
    >>> nrn = ezy.Neuron('some/data/path/morph_file.swc')
    >>> apical_seg_lengths = nrn.get_segment_lengths(ezy.TreeType.apical_dendrite)
    >>> axon_sec_lengths = nrn.get_section_lengths(ezy.TreeType.axon)


- Visualize a neuronal morphology:

.. code-block:: python

    >>> # Initialize nrn as above
    >>> fig, ax = nrn.view()
    >>> fig.show()
