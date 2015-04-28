.. _my-reference-quick-start-guide:

NeuroM Quick Start Guide
**************************

Introduction
=============

NeuroM.
++++++++

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies. 

This quick start guide contains information on how to install NeuroM and start using it. For more detailed information on NeuroM practice, see :ref:`my-reference-tutorial`. 

Installation Manual
===================

Prerequisites
-------------

Operating Systems
++++++++++++++++++

**Windows:**
	To be tested!

**Linux:**
	No specific version or model requirements are known yet. 

**Mac:**
	No specific version or model requirements are known yet.

.. _my-reference-requirements:

Python version and module requirements
++++++++++++++++++++++++++++++++++++++

NeuroM is a Python-based toolkit. In order to use NeuroM, Python 2.7 or later is required. 
In addition, the following Python modules are required:

::

    sys
    os
    scipy
    numpy
    matplotlib

.. _my-reference-installation:

Installation procedure
----------------------

Linux
+++++

.. _my-reference-python-and-required-modules:

Installation of Python and required modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install or upgrade Python.

The first step for a successful installation of NeuroM is the installation of Python and its
most frequently used modules. NeuroM needs at least Python 2.7, so if you have an
earlier version you should upgrade it. If you do not have Python you can download and install 
it from http://www.python.org/getit/.

2. Install individual Python modules.

For NeuroM to work properly, you will need some Python modules to be installed, see :ref:`my-reference-requirements`. 

To check if a module is already installed when Python and IPython installation is complete, 
you can try:

::

    ipython

    import wx, sys, os, shutil, h5py, scipy, numpy, matplotlib, pylab, sklearn

If any of these modules are not already installed, you would get an error "No module named \*".
You have to install the missing modules, before you will be able to run NeuroM. For the 
installation of the above modules and more details, see: http://docs.python.org/2/install/.

Installation of NeuroM
~~~~~~~~~~~~~~~~~~~~~~~~

Once the modules listed in :ref:`my-reference-python-and-required-modules` are downloaded, installed and running, you can start the NeuroM installation procedure. 

1. Git get NeuroM code

- Create a folder to store NeuroM code:

::

    mkdir -p your_git_directory/NeuroM
    cd your_git_directory/NeuroM

- Get NeuroM code out of git repository:

::

    git clone https://bbpteam.epfl.ch/reps/analysis/NeuroM.git

For more information about git, visit: https://git.wiki.kernel.org/index.php/GitSvnCrashCourse.

2. Add NeuroM path to Python path

- In order to be able to run NeuroM from every directory, the NeuroM path should be added to the regular Python path:

::

    cd ~
    nano .bashrc

- Add the following path at the end of the file:

::

    # To add NeuroM to Python path:
    export PYTHONPATH=$PYTHONPATH:~/git/NeuroM

- Save the file and exit:

::

    Control-o
    [return]
    Control-x

NeuroM should be ready to use. 

Tests
~~~~~

In order to verify that the installation was completed successfully, 
you should make sure that it loads correctly and that the tests designed 
to check NeuroM are successful. 

- Load NeuroM and test initialization

At a new terminal you can now test NeuroM. Open a new Python session at the directory of your preference.

.. warning::

   You cannot run NeuroM in the directory: your_git_directory/NeuroM/neurom/. 
   Python will crash in this directory!

Load NeuroM in interactive Python:
   
::

    ipython --pylab
    import neurom

If there are no error messages NeuroM is successfully installed and imported.

- NeuroM check (The tests are not yet ready--)

For validation of the installation and functioning of NeuroM, you can run the tests.
Change to the directory where NeuroM is installed and run the following command:

::

    cd your_git_directory/NeuroM
    nosetests -v --nocapture tests/tools_test.py

Simple example
===============

A simple example to start using NeuroM is illustrated in the following section. 
This example introduces a basic morphological analysis of a single neuron. The user can 
learn how to load the morphology in IPython, view the morphology and create a small report,
that contains the most significant measurements, acording to neuromorpho measurements.
For more detailed information on NeuroM practice, see :ref:`my-reference-tutorial`. 

- Change to your git directory:

::

   cd your_git_directory/

- Enter a Python interactive session from the terminal:

::

   ipython --pylab

- Import NeuroM and start using it:

::

   import neurom as pn

   # Load a population of neurons
   my_population = pn.io.load('/NeuroM/tests/io_test_data/bbp_h5/')

   # Select a neuron from the population to analyze
   my_neuron = my_population.neuron[0]

   # Load a single neuron from the directory that contains multiple files
   my_neuron = pn.io.load('/NeuroM/tests/io_test_data/bbp_h5/C140300B-P1.h5')

- View a single neuronal morphology:

::

   # View the neuron in the xy plane 
   my_neuron.view( plane = 'xy' )


