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

.. _validation:

Morphology validation
*********************

What morphology is valid or invalid? NeuroM completely follows MorphIO in this question because
NeuroM uses MorphIO for reading/writing of morphologies. The rule is be less rigid as possible.
If there is a problem with morphology then NeuroM rather print a warning about instead of raising
an error. If you want validate morphologies as strictly as possible then

.. testcode:: [validation]

   import morphio
   morphio.set_raise_warnings(True)

This will make MorphIO (hence NeuroM as well) raise warnings as errors. You might want to skip some
warnings at all. For example, zero diameter is ok to have in your morpology. Then you can:

.. testcode:: [validation]

   try:
       morphio.set_raise_warnings(True)
       # warnings you are not interested in
       morphio.set_ignored_warning(morphio.Warning.zero_diameter, True)
       m = morphio.Morphology('tests/data/swc/soma_zero_radius.swc')
   finally:
       morphio.set_ignored_warning(morphio.Warning.zero_diameter, False)
       morphio.set_raise_warnings(False)

For more documentation on that topic refer to `<https://morphio.readthedocs.io/en/latest/warnings.html>`__.
