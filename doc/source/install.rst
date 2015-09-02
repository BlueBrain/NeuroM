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


