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



Dependencies
============

Build and runtime
-----------------

.. _pre-dep-label:

``NeuroM`` requires Python version 2.7 or higher.
When installed using `pip <https://pip.pypa.io/en/stable/>`_, ``NeuroM``
will take care of installing unmet dependencies, although it is also possible
to pre-install before ``NeuroM``.

* `numpy <http://www.numpy.org/>`_ >= 1.8.0
* `scipy <http://www.scipy.org/>`_ >= 0.13.3
* `matplotlib <http://www.matplotlib.org/>`_ >= 1.3.1
* `h5py <http://www.h5py.org/>`_ >= 2.2.1
* `enum34 <https://pypi.python.org/pypi/enum34/>`_ >= 1.0.4
* `pyyaml <http://www.pyyaml.org/>`_ >= 3.10.0
* `tqdm <https://pypi.python.org/pypi/tqdm/>`_ tqdm >= 4.8.4


Installing and building
-----------------------

* `pip <https://pip.pypa.io/en/stable/>`_ version 8.1.0 or higher.
* `virtualenv <https://virtualenv.pypa.io/en/stable/>`_

Testing and documentation
-------------------------

These dependencies are not needed for installing and running ``NeuroM``,
but are useful for those who want to contribute to its development.

* `GNU Make <https://www.gnu.org/software/make/>`_
* `nose <https://nose.readthedocs.org/en/latest/>`_
* `coverage <https://coverage.readthedocs.org/en/latest/>`_
* `sphinx <http://sphinx-doc.org/>`_
