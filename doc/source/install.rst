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

.. _installation-label:

Installation
============

It is recommended that you use `pip <https://pip.pypa.io/en/stable/>`_ version 8.1.0
or higher to install into
``NeuroM`` into a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_. For details on
how to set it up, see :ref:`venv-label`

Once the virtualenv is set up, there are three ways to install ``NeuroM``:

#. From the official Python Package Index server (PyPI)
#. From the git repository
#. From source (for NeuroM developers)

.. note::

    In following code samples, the prompts ``(nrm)$`` and ``$`` are used to indicate
    that the user virtualenv is *activated* or *deactivated* respectively.

.. _venv-label:

Virtualenv setup
^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ virtualenv nrm           # creates a virtualenv called "nrm" in the current directory
    $ source nrm/bin/activate  # activates the "nrm" virtualenv
    (nrm)$                     # now we are in the nrm virtualenv

Upgrade the ``pip`` version as shown below:

.. code-block:: bash

    (nrm)$ pip install --upgrade pip   # Install newest pip inside virtualenv if version too old.

To de-activate the virtualenv run the ``deactivate`` command:

.. code-block:: bash

    (nrm)$ deactivate

Note that you do not have to work in the ``nrm`` directory. This is where python
packages will get installed, but you can work anywhere on your file system, as long as
you have activated the ``virtualenv``.

Installation options
^^^^^^^^^^^^^^^^^^^^

Install from the official PyPI server
-------------------------------------

Install the latest release:

.. code-block:: bash

    (nrm)$ pip install neurom

Install a specific version:

.. code-block:: bash

    (nrm)$ pip install neurom==1.2.3

Install from git
----------------

Install a particular release:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git@v0.1.0

Install the latest version:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git


Install from source
-------------------

Clone the repository and install it:

.. code-block:: bash

    (nrm)$ git clone https://github.com/BlueBrain/NeuroM.git
    (nrm)$ pip install -e ./NeuroM

This installs ``NeuroM`` into your ``virtualenv`` in "editable" mode. That means
that changes made to the source code after the installation procedure are seen by the
installed package. To install in read-only mode, omit the ``-e``.
