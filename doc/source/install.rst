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

Note that you do not have to work in the ``nrm`` directory. This is where python
packages will get installed, but you can work anywhere on your file system, as long as
you have activated the ``virtualenv``.

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

Install from git
^^^^^^^^^^^^^^^^

Install a particular release:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git@neurom-v0.0.8

Install the latest version:

.. code-block:: bash

    (nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git


Install from source
^^^^^^^^^^^^^^^^^^^

Clone the repository and install it:

.. code-block:: bash

    (nrm)$ git clone https://github.com/BlueBrain/NeuroM.git
    (nrm)$ pip install -e ./NeuroM

This installs ``NeuroM`` into your ``virtualenv`` in "editable" mode. That means
that changes made to the source code after the installation procedure are seen by the
installed package. To install in read-only mode, omit the ``-e``.

Install from the BBP PyPI server (restricted access)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the latest release:

.. code-block:: bash

    (nrm)$ pip install -i http://bbpgb019.epfl.ch:9090/simple neurom

Install a specific version:

.. code-block:: bash

    (nrm)$ pip install -i http://bbpgb019.epfl.ch:9090/simple neurom==1.2.3

.. warning::

    If your version of pip 7.0 or higher, you need to add the option
    ``--trusted-host bbpgb019.epfl.ch``.

