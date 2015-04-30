NeuroM Quick Start Guide
************************

Introduction
============

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies.

This quick start guide contains information on how to install NeuroM and use case examples.

Dependencies
============

The build-time and runtime dependencies of NeuroM are:

* numpy
* scipy
* matplotlib

Additional dependencies needed for testing and building documentation are

* nose
* coverage
* sphinx
* GNU Make

Installation
============

It is recommended that you use pip to install into ``NeuroM`` into a ``virtualenv``:

.. code-block:: bash

    $ virtualenv --system-site-packages foo   # creates a virtualenv called "foo" in foo directory
    $ source foo/bin/activate                 # activates virtualenv
    (foo)$                                    # now we are in the foo virtualenv

Here, the ``--system-site-packages`` option has been used. This is because dependencies such as
``matplotlib`` aren't trivial to build in a ``virtualenv``. This setting allows python packages
installed in the system to be used inside the ``virtualenv``.

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

The tests require that you have cloned the repository, since the test code is not distributed in the package. It is recommended to use ``nosetests`` for this. There are two options:

Run the tests in a dedicated virtualenv:

.. code-block:: bash

        $ make test

``make`` also has targets for running pylint and pep8:


.. code-block:: bash

        $ make lint       # runs pep8 and pylint if that succeeds
        $ make run_pep8   # run only pep8
        $ make run_pylint # run only pep8

Alternatively, inside the virtualenv, install ``nose`` and ``coverage`` if you haven't
done so already or these aren't installed in the system:

.. code-block:: bash

    (foo)$ pip install nose
    (foo)$ pip install coverage
    (foo)$ nosetests -s -v --with-coverage --cover-package neurom

Building the Documentation
--------------------------

There's  a ``make`` target to build the HTML version of the documentation:

.. code-block:: bash

        $ make doc

This builds the documentation in ``doc/build``.
To view it, point a browser at ``doc/build/html/index.html``

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
