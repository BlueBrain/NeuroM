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

Migration guides
=======================

.. _migration-v3.0.0:

Migration to v3 version
-----------------------

- ``neurom.view.viewer`` is deprecated. To get the same results as before, use the replacement:

   .. code-block:: python

      import neurom as nm
      # instead of: from neurom import viewer
      from neurom.view import matplotlib_impl, matplotlib_utils
      m = nm.load_morphology('some/data/path/morph_file.asc')

      # instead of: viewer.draw(m)
      matplotlib_impl.plot_morph(m)

      # instead of: viewer.draw(m, mode='3d')
      matplotlib_impl.plot_morph3d(m)

      # instead of: viewer.draw(m, mode='dendrogram')
      matplotlib_impl.plot_dendrogram(m)

      # If you used ``output_path`` with any of functions above then the solution is:
      fig, ax = matplotlib_utils.get_figure()
      matplotlib_impl.plot_dendrogram(m, ax)
      matplotlib_utils.plot_style(fig=fig, ax=ax)
      matplotlib_utils.save_plot(fig=fig, output_path=output_path)
      # for other plots like `plot_morph` it is the same, you just need to call `plot_morph` instead
      # of `plot_dendrogram`.

      # instead of `plotly.draw`
      from neurom import plotly_impl
      plotly_impl.plot_morph(m)  # for 2d
      plotly_impl.plot_morph3d(m)  # for 3d

- breaking features changes:
   - use `max_radial_distance` instead of `max_radial_distances`
   - use `number_of_segments` instead of `n_segments`
   - use `number_of_neurites` instead of `n_neurites`
   - use `number_of_sections` instead of `n_sections`
   - use `number_of_bifurcations` instead of `n_bifurcation_points`
   - use `number_of_forking_points` instead of `n_forking_points`
   - use `number_of_leaves` instead of `number_of_terminations`, `n_leaves`
   - use `soma_radius` instead of `soma_radii`
   - use `soma_surface_area` instead of `soma_surface_areas`
   - use `soma_volume` instead of `soma_volumes`
   - use `total_length_per_neurite` instead of `neurite_lengths`
   - use `total_volume_per_neurite` instead of `neurite_volumes`
   - use `terminal_path_lengths` instead of `terminal_path_lengths_per_neurite`
   - use `bifurcation_partitions` instead of `partition`
   - new neurite feature `total_area` that complements `total_area_per_neurite`
   - new neurite feature `volume_density` that complements `neurite_volume_density`


Migration to v2 version
-----------------------
.. _migration-v2:

- ``Neuron`` object now extends ``morphio.Morphology``.
- NeuroM does not remove unifurcations on load. Unifurcation is a section with a single child. Such
  sections are possible in H5 and ASC formats. Now, in order to remove them on your morphology, you
  would need to call ``remove_unifurcations()`` right after the morphology is constructed.

  .. code-block:: python

      import neurom as nm
      nrn = nm.load_morphology('some/data/path/morph_file.asc')
      nrn.remove_unifurcations()

- Soma is not considered as a section anymore. Soma is skipped when iterating over morphology's
  sections. It means that section indexing offset needs to be adjusted by
  ``-(number of soma sections)`` which is usually ``-1``.
- drop ``benchmarks``
- drop ``neurom.check.structural_checks`` as MorphIO does not allow to load invalid morphologies,
  and it does not give access to raw data.
- drop ``Tree`` class. Use ``Section`` instead as it includes its functionality but if you need
  ``Tree`` separately then copy-paste ``Tree`` code from v1 version to your project.
- ``Section`` and ``Neurite`` class can't be copied anymore because their underlying MorphIO
  objects can't be copied (pickled). Only copying of ``Neuron`` is preserved.
- drop ``FstNeuron``. It functionality is included in ``Neuron`` class. Use ``Neuron`` instead of
  ``FstNeuron``.
- Validation of morphologies changed.
    The following is not an invalid morphology anymore:

    - 2 point soma
    - non-sequential ids
- script ``morph_check`` and ``morph_stats`` changed to ``neurom check`` and ``neurom stats``
    correspondingly.