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

Build-time and runtime
^^^^^^^^^^^^^^^^^^^^^^

* `numpy <http://www.numpy.org/>`_
* `h5py <http://www.h5py.org/>`_
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://www.matplotlib.org/>`_
* `enum34 <https://pypi.python.org/pypi/enum34/>`_

It is highly recommended that all except ``enum34`` be installed into your system
before attempting to install ``NeuroM``. The ``NeuroM`` package installation
takes care of installing ``enum34``.

Installing and building
^^^^^^^^^^^^^^^^^^^^^^^

* `pip <https://pip.pypa.io/en/stable/>`_
* `virtualenv <https://virtualenv.pypa.io/en/stable/>`_
* `GNU Make <https://www.gnu.org/software/make/>`_

Testing and documentation
^^^^^^^^^^^^^^^^^^^^^^^^^

These dependencies are not needed for installing and running ``NeuroM``,
but are useful for those who want to contribute to its development.

* `nose <https://nose.readthedocs.org/en/latest/>`_
* `coverage <https://coverage.readthedocs.org/en/latest/>`_
* `sphinx <http://sphinx-doc.org/>`_
