===============
NeuroM Tutorial
===============

1. Installation instructions
============================

`NeuroM <http://neurom.readthedocs.io/en/latest/index.html>`__ is a
Python-based toolkit for the analysis and processing of neuron
morphologies. It is supported for Linux and OS X. Windows users are
advised to use `VirtualBox <https://www.virtualbox.org/>`__. More
detailed installation instructions can be found
`here <http://neurom.readthedocs.io/en/latest/install.html>`__.

1.1 Requirements
----------------

It is assumed that the following packages are installed on your system:

- Python >= 2.7
- pip >= 8.1.0
- ipython
- virtualenv

These are available as packages in most Linux distributions. For OS X,
please refer to `MacPorts <http://www.macports.org/>`__ if you don't
already use a different package manager.

The following Python packages will be installed automatically with
NeuroM if not pre-installed in your system:

- numpy >= 1.8.0
- scipy >= 0.17.0
- h5py >= 2.2.1 (optional)
- matplotlib >= 1.3.1
- pyyaml >= 3.10
- enum34 >= 1.0.4
- tqdm >= 4.8.4
- future >= 0.16.0
- pylru >= 1.0

1.2 Virtual environment set-up
------------------------------

It is recommended that you install NeuroM into a virtual environment.

::

    $ virtualenv nrm                           # creates a virtualenv called "nrm" in nrm directory
    $ source nrm/bin/activate                  # activates virtualenv
    (nrm)$                                     # now we are in the nrm virtualenv

The prompt indicates that the virtual environment has been activated. To
de-activate it:

::

    (nrm)$ deactivate


1.3 Installation from source
----------------------------

Clone the NeuroM repository and install it:

::

    (nrm)$ git clone https://github.com/BlueBrain/NeuroM.git
    (nrm)$ pip install --upgrade pip           # install newest pip inside virtualenv if version too old
    (nrm)$ pip install -e ./NeuroM             # the -e flag makes source changes immediately effective

2. Applications using NeuroM
============================

NeuroM ships with configurable command line applications for commonly
needed functionality.

2.1 morph_check: check the validity of a morphology file
--------------------------------------------------------

The application
`morph_check <http://neurom.readthedocs.io/en/latest/morph_check.html>`__
allows you to apply semantic checks to a morphology file before loading
it into NeuroM:

::

    (nrm)$ morph_check -h                      # shows help for morphology checking script

Try it yourself! You can go to
`NeuroMorpho.Org <http://neuromorpho.org>`__ to download a neuronal
morphology and perform the semantic checks:

::

    (nrm)$ morph_check  path/to/files/filename

2.2 morph_stats: extract basic morphometrics of a sample morphology
-------------------------------------------------------------------

The application
`morph_stats <http://neurom.readthedocs.io/en/latest/morph_stats.html>`__
extracts various morphometrics for one or many morphologies. Its
contents can be easily configured via a configuration file, as shown in
the `online
documentation <http://neurom.readthedocs.io/en/latest/morph_stats.html>`__.

::

    (nrm)$ morph_stats -h                      # shows help for the morphometrics extraction script
    (nrm)$ morph_stats path/to/files/filename  # analyze single morphology file
    (nrm)$ morph_stats path/to/files           # analyze many morphology files

3. The NeuroM Tutorial Notebook
===============================

In the NeuroM repository, you will find a folder ``tutorial``, which
contains a tutorial notebook on NeuroM. For a detailed explanation on
installing and running Jupyter/IPython notebooks, we refer to `the
Jupyter/IPython Notebook Quick Start
Guide <https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/>`__.

First, install ``jupyter`` in the virtual environment. Second, launch
the Jupyter Notebook App. Make sure that you launch the application from
a folder that contains the NeuroM Tutorial notebook.

::

    (nrm)$ pip install jupyter
    (nrm)$ cd /path/to/dir/containing/notebook
    (nrm)$ jupyter notebook                    # launch the Jupyter Notebook App

Next, you can select the notebook that you want to open. Now, you can go
through the tutorial and learn about loading, viewing, and analyzing
neuronal morphologies!
