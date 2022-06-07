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

Examples
========

.. note::

    In following code samples, the prompt ``>>>`` indicates a python interpreter session
    started *with the virtualenv activated*. That gives access to the ``neurom``
    installation.

Analysis with :py:mod:`neurom`
******************************

Here we load a morphology and obtain some information from it:

.. doctest:: [examples]

    >>> import neurom as nm
    >>> m = nm.load_morphology("tests/data/swc/Neuron.swc")
    >>> ap_seg_len = nm.get('segment_lengths', m, neurite_type=nm.APICAL_DENDRITE)
    >>> ax_sec_len = nm.get('section_lengths', m, neurite_type=nm.AXON)


Morphology visualization with the :py:mod:`neurom.view` module
**************************************************************

Here we visualize a morphology:


.. doctest:: [examples]

    >>> # Initialize m as above
    >>> from neurom.view import plot_morph, plot_morph3d, plot_dendrogram
    >>> plot_morph(m)
    >>> plot_morph3d(m)
    >>> plot_dendrogram(m)

Advanced iterator-based feature extraction example
**************************************************

These slightly more complex examples illustrate what can be done with the ``neurom``
module's various generic iterators and simple morphometric functions.

The idea here is that there is a great deal of flexibility to build new analyses based
on some limited number of orthogonal iterator and morphometric components that can
be combined in many ways. Users with some knowledge of ``python`` and ``neurom`` can easily
implement code to obtain new morphometrics.

All of the examples in the previous sections can be implemented
in a similar way to those presented here.


.. literalinclude:: ../../examples/iteration_analysis.py
    :lines: 30-

Getting Log Information
***********************

``neurom`` emits many logging statements during the course of its functioning.
They are emitted in the ``neurom`` namespace, and can thus be filtered based
on this.  An example of setting up a handler is:

.. doctest::

    >>> import logging
    >>> # setup which namespace will be examined, and at what level
    >>> # in this case we only want messages from 'neurom' and all messages
    >>> # (ie: DEBUG, INFO, etc)
    >>> logger = logging.getLogger('neurom')
    >>> logger.setLevel(logging.DEBUG)
    >>> # setup where the output will be saved, in this case the console
    >>> sh = logging.StreamHandler()
    >>> logger.addHandler(sh)

For more information on logging, it is recommended to read the official Python
logging HOWTOs: `Python 3 <https://docs.python.org/3/howto/logging.html>`_.
