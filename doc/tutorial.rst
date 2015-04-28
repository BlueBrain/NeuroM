.. _my-reference-tutorial:

NeuroM Tutorial
*****************

Introduction
=============

NeuroM stands for \ **P**\ ython \ **Neu**\ ronal \ **M**\ orphology \ **A**\ nalysis \ **T**\ ool\ **k**\ it.

NeuroM is a Python-based toolkit for the analysis and processing of neuron morphologies. 

This tutorial will take you through the basic applications of NeuroM. 
For instructions the installation process of NeuroM see :ref:`my-reference-quick-start-guide`. 

NeuroM provides basic Python tools to visualize neuronal morphologies, extract morphological features, 
plot those features to explore their behavior and any correlations between them. NeuroM provides 
useful reports as a summary for ....  -----TODO------ Add overview of NeuroM code

The main morphological entities that can be analyzed with NeuroM is a Tree, a Neuron and a Population (coming soon...) .
A Neuron consists of a Soma and a set of neuronal Trees (axons, basal and apical dendrites) that are represented as
independent objects in NeuroM.

Table of contents:

1. :ref:`my-reference-io`

  * :ref:`my-reference-io-load`

2. :ref:`my-reference-single-neuron`

  * :ref:`my-reference-single-neuron-view`
  * :ref:`my-reference-single-neuron-analyze`
  * :ref:`my-reference-single-neuron-plot-analysis`
  * :ref:`my-reference-single-neuron-report`

3. :ref:`my-reference-population`

  * :ref:`my-reference-population-view`
  * :ref:`my-reference-population-analyze`
  * :ref:`my-reference-population-plot-analysis`
  * :ref:`my-reference-population-fit-analysis`
  * :ref:`my-reference-population-report`

4. :ref:`my-reference-population-comparison`

  * :ref:`my-reference-population-compare-morphologies`
  * :ref:`my-reference-population-compare-statistics`
  * :ref:`my-reference-population-compare-neuron-population`
  * :ref:`my-reference-population-compare-multiple-populations`

IO
====


Basic Use Instructions
-------------------------

In order to use the sample morphologies, that are provided, you should go to your_git_directory/NeuroM/. 
The sample morphologies are in your_git_directory/NeuroM/tests/io_test_data (*To be confirmed in final version*). 
Change your current directory to the main NeuroM directory:

::

   cd your_git_directory/NeuroM/

The first step for an anatomical analysis of a morphology is to load the corresponding files in Python. 
NeuroM offers a useful io-tool for loading and saving morphologies in different file formats. 
It also provides the possibility to convert between the supported file formats. 
The current morphology file formats supported by NeuroM (*To be confirmed in final version*) are:

::

   swc
   ascii (coming not so soon...)
   xml   (coming not so soon...)
   h5    (coming not so soon...)

Open an interactive Python module in a terminal:

::

   ipython --pylab

Import NeuroM module in Python:

::

   # To import NeuroM
   import neurom as pn

.. _my-reference-io-load:

Load morphologies
-----------------

NeuroM supports the loading of a single morphology (:ref:`my-reference-load-neuron`), 
of a set of morphologies (:ref:`my-reference-load-population`), or a superset of morphologies (:ref:`my-reference-load-populationset`).
The sample morphology files that NeuroM provides for test purposes can be found in 'NeuroM/tests/io_test_data'.

.. _my-reference-load-neuron:

Neuron
^^^^^^^^

::

   # To load a single neuron the path and the filename of the neuron should be specified.
   my_neuron = pn.io.load('tests/...') 

.. _my-reference-load-population:

Population
^^^^^^^^^^^

::

   # To load a population of neurons, in .swc format, only the directory that contains 
   # the set of files should be provided. All contained morphologies will be loaded 
   # as a Population object.
   my_population = pn.io.load('tests/io_test_data/bbp_h5/') 


.. _my-reference-io-save:

Save morphologies
------------------

NeuroM provides the possibility to modify a morphology. For this reason, 
it is also useful to save the modified morphology for future use, using the 
io.save tool of NeuroM. The default save format is .swc format. 

.. _my-reference-save-neuron:

Neuron
^^^^^^^^

::

   # A single morphology can be saved in the default .swc format.
   pn.io.save(my_neuron)
   # The neuron's initial name will be used as the file name, if not defined.
   # Alternatively, the neuron can be saved, as follows:
   my_neuron.save()
   # The neuron can be saved in any of the provided file formats
   my_neuron.save(output_format='swc')
   pn.io.advanced.save_swc(my_neuron)

::

   # A different name can be selected for the neuron file:
   pn.io.save(my_neuron, output_name='my_new_neuron')
   my_neuron.save(output_name='my_new_neuron')
   # Or a different path for the neuron to be saved:
   my_neuron.save(output_path='./path/to/save/neuron')


Single Neuron
=============

The morphology of a neuron is the key factor for studying
the general principles of structural organization in the brain.
A single neuronal morphology can be viewed, analyzed and studied in details,
through the different NeuroM modules:


  * :ref:`my-reference-single-neuron-view`
  * :ref:`my-reference-single-neuron-analyze`
  * :ref:`my-reference-single-neuron-plot-analysis`
  * :ref:`my-reference-single-neuron-report`

In order to analyze a Neuronal morphology, you first need to load the morphology in Python, 
see :ref:`my-reference-load-neuron`. For the following section, we will use the loaded morphology 
from NeuroM directory 'my_neuron'. All plots of this section are produced from the mentioned Neuron
and can be reproduced by using the corresponding code, for each test case.

.. _my-reference-single-neuron-view:

View
------

View a Neuron
^^^^^^^^^^^^^

::

   # To view the 2d projection of a neuron:
   my_neuron.view()
   pn.view.neuron(my_neuron)


.. figure:: /_static/tutorial_figures/neuron_1.png
   :scale: 30 %

   Sample 2d view figure of a neuron morphology.

View 3d
^^^^^^^^

::

   # To view the 3d morphology of a neuron:
   my_neuron.view3d()


.. figure:: /_static/tutorial_figures/neuron_2.png
   :scale: 30 %

   Sample 3d view figure of a neuron morphology.


Analyze
--------

Extract selected morphomrtrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   # To get the values of a selected morphological feature
   # of all the fragments of a Neuron:
   my_neuron.get_fragment_lengths()
   # To get the values of a selected morphological feature
   # of all the sections of a Neuron:
   my_neuron.get_section_lengths()
   # To get the total value of a selected morphological feature
   # of a Neuron:
   my_neuron.get_neuron_length()

.. _my-reference-single-neuron-feature-list:

Feature list
^^^^^^^^^^^^^

The complete list of features that are provided by NeuroM at the level of a single Neuron analysis,
can be summarized in the following table:

+-------------------------------+---------------------------+----------------------------------+----------------------+
| Section features              | Fragment features         | Trunk features                   | Overall features     |
+===============================+===========================+==================================+======================+
| section_lengths               | fragment_lengths          | trunk_length                     | neuron_length        |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_surface_areas         | fragment_surface_areas    | trunk_surface_area               | neuron_surface_area  |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_volumes               | fragment_volumes          | trunk_volume                     | neuron_volume        |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_x                     | fragment_central_x        | trunk_azimuth_from_center        | number_trees         |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_y                     | fragment_central_y        | trunk_vector_to_centroid         | soma_cs_area         |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_z                     | fragment_central_z        | trunk_distal_vector_to_centroid  | soma_diameter        |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_branch_orders         | fragment_branch_orders    | trunk_distance_from_centroid     | bounding_box         |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_path_distances        | fragment_path_distances   | trunk_elevation_from_center      | fractal_dimensions   |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_radial_distances      | fragment_radial_distances | trunk_diameters                  | sholl_analysis       |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_number                | fragment_diameters        | trunk_distal_diameter            |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_hs_orders             | fragment_hs_orders        |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_tortuosities          | fragment_summed_length    |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_contractions          | fragment_meander_angles   |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_daughter_ratio        | fragment_coordinates      |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_length_main_apical    | fragment_taper_rate       |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_rall_ratio            | fragment_unit_vectors     |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_bif_lengths           |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_bif_path_distances    |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_bif_radial_distances  |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_bif_angles_local      |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_bif_angles_remote     |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_term_lengths          |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_term_path_distances   |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_term_radial_distances |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_partition             |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+
| section_partition_asymmetry   |                           |                                  |                      |
+-------------------------------+---------------------------+----------------------------------+----------------------+


.. _my-reference-single-neuron-plot-analysis:

Plot Analysis
--------------

Histogram
^^^^^^^^^^^^

::

   # To plot the histogram of a selected morphological feature
   # for all the fragments of a Neuron:
   pn.plot.histogram(my_neuron, feature='fragment_lengths')
   # To plot the histogram of a selected morphological feature
   # for all the sections of a Neuron:
   pn.plot.histogram(my_neuron, feature='section_lengths')

.. figure:: /_static/tutorial_figures/neuron_7.png
   :scale: 30 %

   Histogram of section lengths of a neuron.

Boxplot
^^^^^^^^^

::

   # To plot the boxplot of a selected morphological feature
   # for all the fragments of a Neuron:
   pn.plot.boxplot(my_neuron, feature='fragment_lengths')
   # To plot the boxplot of a selected morphological feature
   # for all the sections of a Neuron:
   pn.plot.boxplot(my_neuron, feature='section_lengths')


.. figure:: /_static/tutorial_figures/neuron_9.png
   :scale: 30 %

   Boxplot of section lengths of a neuron.


Report Statistics
------------------

Neuromorpho report
^^^^^^^^^^^^^^^^^^^
::

   # To generate a report that contains a view of the Neuron's morphology
   # and its basic morphometrics, that correspond to the list of measurements
   # that neuronmorpho provides:
   pn.report.neuromorpho(my_neuron, output_path='selected_directory')

