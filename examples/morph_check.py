#!/usr/bin/env python

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

'''Examples of basic data checks'''
from neurom.io.utils import load_neuron
from neurom.check import morphology as io_chk
from neurom.exceptions import SomaError
from neurom.exceptions import NonConsecutiveIDsError
import argparse
import os

DESCRIPTION = '''
NeuroM Morphology Checker
=========================
'''

EPILOG = '''
Description
-----------

Performs checks on reconstructed morphologies frim data contained in
morphology files.

Files are unloaded into neuron objects before testing. This means they must
have a soma and no format errors.

Errors checked for
------------------
* No soma
* No axon
* No apical dendrite
* No basal dendrite
* Zero radius points
* Zero length segments
* Zero length sections

Examples
--------
morph_check.py --help               # print this help
morph_check.py some/path/neuron.h5  # Process an HDF5 file
morph_check.py some/path/neuron.swc # Process an SWC file
morph_check.py some/path/           # Process all HDF5 and SWC files found in directory
'''


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EPILOG)
    parser.add_argument('datapath',
                        help='Path to morphology data file or directory')

    return parser.parse_args()


def get_morph_files(directory):
    '''Get a list of all morphology files in directory dir

    Returns files with extensions '.swc' or '.h5' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return [m for m in lsdir
            if os.path.isfile(m) and
            os.path.splitext(m)[1].lower() in ('.swc', '.h5')]


def test_file(f):
    '''Run tests on a morphology file'''
    print '\nCheck file %s...' % f
    try:
        nrn = load_neuron(f)
        print 'Has valid soma? True'
    except SomaError:
        print 'Has valid soma? False'
        print 'Cannot continue without a soma... Aborting'
        return

    print 'Has axon? %s' % io_chk.has_axon(nrn)
    print 'Has apical dendrite? %s' % io_chk.has_apical_dendrite(nrn)
    print 'Has basal dendrite? %s' % io_chk.has_basal_dendrite(nrn)

    fr = io_chk.all_nonzero_neurite_radii(nrn)
    print 'All neurites have non-zero radius? %s' % fr[0]
    if not fr[0]:
        print '%s points with zero radius detected: %s' % (len(fr[1]), fr[1])

    fs = io_chk.all_nonzero_segment_lengths(nrn)
    print 'All segments have non-zero length? %s' % fs[0]
    if not fs[0]:
        print '\tSegments with zero length detected:', fs[1]

    fs = io_chk.all_nonzero_section_lengths(nrn)
    print 'All sections have non-zero length? %s' % fs[0]
    if not fs[0]:
        print '\tSections with zero length detected:', fs[1]


if __name__ == '__main__':

    args = parse_args()
    data_path = args.datapath
    if os.path.isfile(data_path):
        files = [data_path]
    elif os.path.isdir(data_path):
        print 'Checking files in directory', data_path
        files = get_morph_files(data_path)

    for _f in files:
        try:
            test_file(_f)
        except NonConsecutiveIDsError as e:
            print 'ERROR in file %s: %s' % (_f, e.message)
        except StandardError:
            print 'ERROR: Could not read file %s.' % _f
