NeuroM Quick Start Guide
************************

Introduction
============

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies.

This quick start guide contains information on how to install NeuroM and use case examples.

Dependencies
============

The known build-time and runtime dependencies of NeuroM are:

* numpy
* scipy
* matplotlib


Installation
============

It is recommended that you use pip to install into a ``virtualenv``:

.. code-block:: bash

    $ virtualenv --system-site-packages foo   # creates a virtualenv called "foo" in foo directory
    $ source foo/bin/activate                 # activates virtualenv
    (foo)$                                    # now we are in the foo virtualenv

The prompt indicates that the ``virtualenv`` has been activated. To de-activate it,

.. code-block:: bash

    (foo)$ deactivate

Note you do not have to work in the ``foo`` directory. This is where python packages will get installed, but you can work anywhere on your file system, as long as you have activated the ``virtualenv``.

NeuroM
------

Once the ``virtualenv`` is set up, you can install ``hbp-neurom`` from source. First, clone the repository

.. code-block:: bash

    (foo)$ git clone ssh://bbpcode.epfl.ch/algorithms/hbp-neurom
    (foo)$ pip install -e ./hbp-neurom

This installs ``hbp-neurom`` into your ``virtualenv`` in "editable" mode. That means changes you make to the source code are seen by the installation.
To install in read-only mode, omit the ``-e``.


Running the tests
-----------------

The tests require that you have cloned the repository, since the test code is not distributed in the package. It is recommended to use ``nosetests`` for this. Inside the virtualenv, install ``nose`` if you haven't done so already.

.. code-block:: bash

    (foo)$ pip install nose
    (foo)$ nosetests -s -v path_to_repo/hbp-neurom/neurom/tests

Examples
========

- Load a neuron:

.. code-block:: bash

    # Load a neuron
    (foo)$

- Visualize a neuronal morphology:

.. code-block:: bash

    # Visualize a neuronal morphology
    (foo)$

- Abstract morphometrics:

.. code-block:: bash

    # Abstract morphometrics
    (foo)$
