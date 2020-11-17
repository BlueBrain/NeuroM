<!--
 Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
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
 -->
![NeuroM Logo](doc/source/logo/NeuroM.jpg)

# NeuroM

NeuroM is a Python toolkit for the analysis and processing of neuron morphologies.


[![Build Status](https://travis-ci.org/BlueBrain/NeuroM.svg?branch=master)](https://travis-ci.org/BlueBrain/NeuroM)
[![license](https://img.shields.io/pypi/l/neurom.svg)](https://github.com/BlueBrain/NeuroM/blob/master/LICENSE.txt)
[![codecov.io](https://codecov.io/github/BlueBrain/NeuroM/coverage.svg?branch=master)](https://codecov.io/github/BlueBrain/NeuroM?branch=master)
[![Documentation Status](https://readthedocs.org/projects/neurom/badge/?version=latest)](http://neurom.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.209498.svg)](https://doi.org/10.5281/zenodo.209498)

## Acknowledgements

This project/research received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA). 
This research was supported by the EBRAINS research infrastructure, funded from the European
Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant
Agreement No. 945539 (Human Brain Project SGA3).

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

## Documentation

NeuroM documentation is built and hosted on [readthedocs](https://readthedocs.org/).

* [latest snapshot](http://neurom.readthedocs.org/en/latest/)
* [latest release](http://neurom.readthedocs.org/en/stable/)

## Installation

It is recommended that you use [`pip`](https://pip.pypa.io/en/stable/) to install
`NeuroM` into a [`virtualenv`](https://virtualenv.pypa.io/en/stable/). The following
assumes a `virtualenv` named `nrm` has been set up and
activated. We will see three ways to install `NeuroM`


### 1. From the Python Package Index

```
(nrm)$ pip install neurom
```

### 2. From git repository

```
(nrm)$ pip install git+https://github.com/BlueBrain/NeuroM.git
```

### 3. From source

Clone the repository and install it:

```
(nrm)$ git clone https://github.com/BlueBrain/NeuroM.git
(nrm)$ pip install -e ./NeuroM
```

This installs `NeuroM` into your `virtualenv` in "editable" mode. That means changes
made to the source code are seen by the installation. To install in read-only mode, omit
the `-e`.

## Tutorial notebook

To gain an understanding of some of the capabilities of NeuroM, you can launch
the NeuroM tutorial on MyBinder. Click the badge to launch, no need to download or
install!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BlueBrain/NeuroM.git/master?filepath=tutorial%2Fgetting_started.ipynb)

## Examples

- Extract morphometrics from one or many morphology files:

```
$ morph_stats some/path/morph.swc # single file
{
  "some/path/morph.swc":{
    "axon":{
      "total_section_length":207.87975220908129,
      "max_section_length":11.018460736176685,
      "max_section_branch_order":10,
      "total_section_volume":276.73857657289523
    },
    "all":{
      "total_section_length":840.68521442251949,
      "max_section_length":11.758281556059444,
      "max_section_branch_order":10,
      "total_section_volume":1104.9077419665782
    },
    "mean_soma_radius":0.17071067811865476,
    "apical_dendrite":{
      "total_section_length":214.37304577550353,
      "max_section_length":11.758281556059444,
      "max_section_branch_order":10,
      "total_section_volume":271.9412385728449
    },
    "basal_dendrite":{
      "total_section_length":418.43241643793476,
      "max_section_length":11.652508126101711,
      "max_section_branch_order":10,
      "total_section_volume":556.22792682083821
    }
  }
}

$ morph_stats some/path # all files in directory
```

- Perform checks on neuron morphology files:

```
(nrm)$ morph_check some/data/path/morph_file.swc # single file
INFO: ========================================
INFO: File: some/data/path/morph_file.swc
INFO:                      Is single tree PASS
INFO:                     Has soma points PASS
INFO:                  Has sequential ids PASS
INFO:                  Has increasing ids PASS
INFO:                      Has valid soma PASS
INFO:                  Has valid neurites PASS
INFO:                  Has basal dendrite PASS
INFO:                            Has axon PASS
INFO:                 Has apical dendrite PASS
INFO:     Has all nonzero segment lengths PASS
INFO:     Has all nonzero section lengths PASS
INFO:       Has all nonzero neurite radii PASS
INFO:             Has nonzero soma radius PASS
INFO:                                 ALL PASS
INFO: ========================================

(nrm)$ morph_check some/data/path # all files in directory
    ....
```

- Load a neuron and obtain some information from it:

```python
>>> import neurom as nm
>>> nrn = nm.load_neuron('some/data/path/morph_file.swc')
>>> apical_seg_lengths = nm.get('segment_lengths', nrn, neurite_type=nm.APICAL_DENDRITE)
>>> axon_sec_lengths = nm.get('section_lengths', nrn, neurite_type=nm.AXON)
```


- Visualize a neuronal morphology:

```python
>>> # Initialize nrn as above
>>> from neurom import viewer
>>> fig, ax = viewer.draw(nrn)
>>> fig.show()
>>> fig, ax = viewer.draw(nrn, mode='3d') # valid modes '2d', '3d', 'dendrogram'
>>> fig.show()
```


## Dependencies

The build-time and runtime dependencies of NeuroM are:

* [numpy](http://www.numpy.org/)
* [h5py](http://www.h5py.org/)
* [scipy](http://www.scipy.org/)
* [matplotlib](http://www.matplotlib.org/)
* [enum-compat](https://pypi.python.org/pypi/enum-compat/)
* [pyyaml](http://www.pyyaml.org/)


## Reporting issues

Issues should be reported to the
[NeuroM github repository issue tracker](https://github.com/BlueBrain/NeuroM/issues).
The ability and speed with which issues can be resolved depends on how complete and
succinct the report is. For this reason, it is recommended that reports be accompanied
with a minimal but self-contained code sample that reproduces the issue, the observed and
expected output, and if possible, the commit ID of the version used. If reporting a
regression, the commit ID of the change that introduced the problem is also extremely valuable
information.
