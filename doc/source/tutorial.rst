========
Tutorial
========

The NeuroM tutorial notebook
============================

NeuroM includes tutorial notebooks under the :file:`tutorial` subdirectory
of the repository. You can launch the tutorials using MyBinder or with
Jupyter notebooks. Refer to :ref:`installation-label` for help installing.

MyBinder
--------

To launch the tutorial in your browser using MyBinder, click the badge |badge|

.. |badge| image:: https://mybinder.org/badge_logo.svg
              :target: https://mybinder.org/v2/gh/BlueBrain/NeuroM/master?filepath=tutorial%2Fgetting_started.ipynb

Jupyter notebooks
-----------------

For a detailed explanation on installing and running Jupyter/IPython notebooks,
we refer to the `Jupyter/IPython Notebook Quick Start
Guide <https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/>`__.

First, install ``jupyter`` in the virtual environment. Second, launch
the Jupyter Notebook App. Make sure that you launch the application from
a folder that contains the NeuroM Tutorial notebook.

.. code-block:: bash

    (nrm)$ pip install jupyter
    (nrm)$ cd /path/to/dir/containing/notebook
    (nrm)$ jupyter notebook                    # launch the Jupyter Notebook App

Next, you can select the notebook that you want to open. Now, you can go
through the tutorial and learn about loading, viewing, and analyzing
neuronal morphologies!

Applications using NeuroM
=========================

NeuroM ships with configurable command line applications for commonly
needed functionality.

neurom check: check the validity of a morphology file
-----------------------------------------------------

The application
`neurom check <http://neurom.readthedocs.io/en/latest/morph_check.html>`__
allows you to apply semantic checks to a morphology file before loading
it into NeuroM:

.. code-block:: bash

    (nrm)$ neurom check -h                      # shows help for morphology checking script

Try it yourself! You can go to
`NeuroMorpho.Org <http://neuromorpho.org>`__ to download a neuronal
morphology and perform the semantic checks:

.. code-block:: bash

    (nrm)$ neurom check path/to/files/filename

morph_stats: extract basic morphometrics of a sample morphology
---------------------------------------------------------------

The application
`morph_stats <http://neurom.readthedocs.io/en/latest/morph_stats.html>`__
extracts various morphometrics for one or many morphologies. Its
contents can be easily configured via a configuration file, as shown in
the `online
documentation <http://neurom.readthedocs.io/en/latest/morph_stats.html>`__.

.. code-block:: bash

    (nrm)$ neurom stats -h                      # shows help for the morphometrics extraction script
    (nrm)$ neurom stats path/to/files/filename  # analyze single morphology file
    (nrm)$ neurom stats path/to/files           # analyze many morphology files

