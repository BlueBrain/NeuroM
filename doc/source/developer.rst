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

Developer Documentation
=======================

Development Workflow
--------------------

* Fork from github
* Develop on your fork
* Test locally
* Make a pull request

Before making a pull request, make sure that your fork is up to date and that all the
tests pass locally. This will make it less likely that your pull request will get
rejected by making breaking changes or by failing the test requirements.

Running the tests
-----------------

The tests require that you have cloned the repository, since the test code is
not distributed in the package. It is recommended to use ``pytest`` for
this. There are two options:

Use the provided ``Makefile`` to run the tests using ``make``:

.. code-block:: bash

    $ git clone https://github.com/BlueBrain/NeuroM.git
    $ cd NeuroM
    $ make test

This runs ``pep8``, ``pylint`` and the unit tests in sequence.

This method takes care of
installing all extra dependencies needed for running the tests, diagnosing the results,
performing linting on the source code. These dependencies are installed into a
``virtuanelv`` named ``neurom_test_venv``:

The ``Makefile`` also has targets for running only ``pylint`` and ``pep8`` individually:

.. code-block:: bash

        $ make lint       # runs pep8 and pylint if that succeeds
        $ make run_pep8   # run only pep8
        $ make run_pylint # run only pylint


Note that you can also install the test dependencies and run the tests inside of your
own ``virtualenv``:

.. code-block:: bash

    (nrm)$ pip install -r requirements_dev.txt

This installs the following packages into your ``virtualenv``
unless they are already installed:

.. literalinclude:: ../../requirements_dev.txt
    :lines: 29-36

Then, run the tests manually in the ``virtualenv``. For example,

.. code-block:: bash

    (nrm)$ pytest --cov-report term-missing --cov-report xml --cov-report html --cov=neurom neurom

.. warning::

    To ensure that the test requirements are the same as those run in continuous
    integration, you should run the tests using the ``make test`` command. This is the
    same command used in continuous integration. Failure to pass means will result in
    a pull request being rejected.

.. include:: documentation.rst

Python compatibility
--------------------

We test the code against Python 3.6 and 3.7.
