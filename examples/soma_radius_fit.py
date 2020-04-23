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

"""Extract a distribution for the soma radii of the population (list) of neurons.
   for the soma radii of the population (list) of neurons.
   """

import argparse

import neurom as nm
from neurom import stats as st


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Morphology fit distribution extractor',
        epilog='Note: Prints the optimal distribution and corresponding parameters.')

    parser.add_argument('datapath',
                        help='Path to morphology data file or directory')

    return parser.parse_args()


def test_multiple_distr(filepath):
    """Runs the distribution fit for multiple distributions and returns
       the optimal distribution along with the corresponding parameters.
    """
    #  load a neuron from an SWC file
    population = nm.load_neurons(filepath)

    # Create a list of basic distributions
    distr_to_check = ('norm', 'expon', 'uniform')

    # Get the soma radii of a population of neurons
    soma_size = nm.get('soma_radii', population)

    # Find the best fit distribution
    return st.optimal_distribution(soma_size, distr_to_check)


if __name__ == '__main__':
    args = parse_args()

    data_path = args.datapath

    result = test_multiple_distr(data_path)
    print("Optimal distribution fit for soma radius is: %s with parameters %s" %
          (result.type, result.params))
