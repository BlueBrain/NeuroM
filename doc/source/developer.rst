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
not distributed in the package. It is recommended to use ``tox`` for this.

.. code-block:: bash

    $ git clone https://github.com/BlueBrain/NeuroM.git
    $ cd NeuroM
    $ tox

This method takes care of installing all extra dependencies needed for tests, diagnosing results,
etc. It runs documentation check, pylint and the tests in sequence. Also it can run them
individually:

.. code-block:: bash

        $ tox -e py38-lint       # runs only pylint
        $ tox -e py38-docs       # run only documentation check
        $ tox -e py38            # run only the tests

You can also run tests manually via `pylint` but you need to make sure that you have installed
all required dependencies in your virtual environment:

.. code-block:: bash

    (your virtual env name)$ pip install neurom[plotly] pytest

Then, run the tests manually. For example,

.. code-block:: bash

    (your virtual env name)$ pytest tests

.. include:: documentation.rst

Python compatibility
--------------------

We test the code against Python 3.8, 3.9, 3.10, and 3.11.
