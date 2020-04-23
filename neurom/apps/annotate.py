# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Anomaly and artefact detection and annotation generation."""
import logging
from itertools import chain

from neurom.core.dataformat import COLS

L = logging.getLogger(__name__)


def generate_annotation(result, settings):
    """Generate the annotation for a given checker.

    Arguments:
        result: the result of the checker
        settings: the display settings for NeuroLucida
    Returns
        An S-expression-like string representing the annotation
    """
    if result.status:
        return ""

    header = ("\n\n"
              "({label}   ; MUK_ANNOTATION\n"
              "    (Color {color})   ; MUK_ANNOTATION\n"
              "    (Name \"{name}\")   ; MUK_ANNOTATION").format(**settings)

    points = [p for _, _points in result.info for p in _points]

    annotations = ("    ({0} {1} {2} 0.50)   ; MUK_ANNOTATION".format(
        p[COLS.X], p[COLS.Y], p[COLS.Z]) for p in points)
    footer = ")   ; MUK_ANNOTATION\n"

    return '\n'.join(chain.from_iterable(([header], annotations, [footer])))


def annotate(results, settings):
    """Concatenate the annotations of all checkers."""
    annotations = (generate_annotation(result, setting)
                   for result, setting in zip(results, settings))
    return '\n'.join(annot for annot in annotations if annot)
