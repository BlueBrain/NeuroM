Changelog
=========

Version 4.0.0
-------------

- Morphology class accepts only morphio objects, not files anymore. (#1120)
- Replace ``iter_*`` methods by properties in core objects and improve ``iter_segments``. (#1054)
- NeuriteType extended to allow mixed type declarations as tuple of ints. (#1071)
- All features return built-in types (#1064)
- Morphology class also allows mutable morphio objects to be passed explicitly. (#1049)
- Morphology class uses morphio immutable class by composition, istead of inheritance. (#979)
- Morphology level radial distance features use the soma as reference point. (#1030)
- Make ``neurom.core.Population`` resolve paths. Symlinks are not resolved. (#1047)
- Mixed subtree processing can be used in morph_stats app via the use_subtrees flag. (#1034)
- ``neurom.view.[plot_tree|plot_tree3d|plot_soma|plot_soma3D]`` were hidden from the
  neurom.view module. They can still be imported from neurom.view.matplotlib_impl. (#1032)
- Mixed subtree processing. (#981)
- Deprecated modules and classes were removed. (#1026)


Version 3.2.3
-------------

- Fix neurom.app.morph_stats.extract_dataframe for Population objects with several workers (#1080)
- Update readthedocs config (#1077)
- Can pass List[Population] args to neurom.app.morph_stats.extract_dataframe (#1076)
- Fix compatibility with MorphIO>=3.3.6 (#1075)
- Check that soma is not empty when features need it (#1073)

Version 3.2.2
-------------

- Fix QhullError warning (#1063)

Version 3.2.1
-------------

- Fix: extract_stats could not work on single neurite (#1060)
- Fix view cli to use 'equal' axis when available (#1051)
- Remove single point contour somas in h5 and asc tests (#1045)
- Remove duplicated deps jinja, sphinx (#1043)


Version 3.2.0
-------------

- Add ``neurom.features.morphology.length_fraction_above_soma`` feature.
- List of multiple kwargs configurations are now allowed in``neurom.apps.morph_stats``.
- ``neurom.features.neurite.principal_direction_extents`` directions correspond to extents
  ordered in a descending order.
- Add features ``neurom.features.morphology.(aspect_ration, circularity, shape_factor)```
- Fix ``neurom.morphmath.principal_direction_extent`` to calculate correctly the pca extent.
- Fix ``neurom.features.neurite.segment_taper_rates`` to return signed taper rates.
- Fix warning system so that it doesn't change the pre-existing warnings configuration
- Fix ``neurom.features.bifurcation.partition_asymmetry`` Uylings variant to not throw
  for bifurcations with leaves.
- Fix ``neurom.features.neurite.principal_direction_extents`` to remove duplicate points
  when calculated.
- Add ``neurom.features.morphology.volume_density`` feature so that it is calculated
  correctly when the entire morphology is taken into account instead of summing the per
  neurite volume densities.
- Add support for py39 and py310 testing.
- Fix ``neurom.features.morphology.sholl_frequency`` to return an empty list when a
  neurite_type that is not present in the morphology is specified.
- Fix ``neurom.features.morphology.trunk_origin_radii`` to warn and use only the root
  section for the calculation of the path distances. Edge cases from the combination
  of ``min_length_filter`` and ``max_length_filter`` are addressed.
- Fix ``neurom.features.morphology.sholl_frequency`` to use soma center in distance
  calculation, instead of using the origin.
- Add ``neurom.features.morphology.trunk_angles_inter_types`` and
  ``neurom.features.morphology.trunk_angles_from_vector`` features, make
  ``neurom.features.morphology.trunk_angles`` more generic and add length filter to
  ``neurom.features.morphology.trunk_origin_radii``.
- Deprecate python3.6
- Add doc on spherical coordinates.

Version 3.1.0
-------------
- Add morphology features total_width, total_height and total_depth

Version 3.0.2
-------------
- Fix 'raw' mode in ``neurom stats``.
- Add example astrocyte analysis notebook.
- Fix readthedocs documentation build.
- Delete all requirements txt files and update documentation accordingly.
- Adding back unifurcation check

Version 3.0.1
-------------
- Add method to Soma class to check wether it overlaps a point or not.
- Ensure ``neurom.morphmath.angle_between_vectors`` always return 0 when the vectors are equal.

Version 3.0.0
-------------
- Rename all 'neuron' names to 'morphology' names, including module and package names. Previous
  'neuron' names still exist but deprecated. It is recommended to use new names:

    - ``neurom.check.neuron_checks`` => ``neurom.check.morphology_checks``, replace `neuron_checks`
      with `morphology_checks` in configs for ``neurom check``.
    - ``neurom.core.neuron`` => ``neurom.core.morphology``
    - ``neurom.core.neuron.Neuron`` => ``neurom.core.morphology.Morphology``
    - ``neurom.core.neuron.graft_neuron`` => ``neurom.core.morphology.graft_morphology``
    - ``neurom.io.utils.load_neuron`` => ``neurom.io.utils.load_morphology``
    - ``neurom.io.utils.load_neurons`` => ``neurom.io.utils.load_morphologies``
    - ``neurom.core.Population.neurons`` => ``neurom.core.Population.morphologies``

- Refactor plotting functionality. :ref:`migration-v3.0.0`.
    - deprecate ``neurom.view.viewer``
    - rename ``neurom.view.view`` to ``neurom.view.matplotlib_impl``
    - rename ``neurom.view.plotly`` to ``neurom.view.plotly_impl``
    - rename ``neurom.view.common`` to ``neurom.view.matplotlib_utils``
    - swap arguments ``ax`` and ``nrn`` of all plot functions in ``neurom.view.matplotlib_impl``,
      also ``nrn`` arg is renamed to ``morph``.
    - delete ``neurom.view.plotly.draw``. Use instead ``neurom.view.plotly_impl.plot_morph`` and
      ``neurom.view.plotly_impl.plot_morph3d``.

- Refactor features.
    - Drop 'func' suffix of all module names within `features` package:
        - ``neurom.features.bifurcationfunc`` => ``neurom.features.bifurcation``
        - ``neurom.features.sectionfunc`` => ``neurom.features.section``
        - ``neurom.features.neuritefunc`` => ``neurom.features.neurite``
        - ``neurom.features.neuronfunc`` => ``neurom.features.morphology``
    - Rigid classification of features. ``neurite`` features must accept only a single neurite.
      ``morphology`` features must accept only a single morphology. ``population`` features must
      accept only a collection of neurons or a neuron population.
    - Some features were deleted, renamed, added. See :ref:`migration-v3.0.0`.
    - Name consistency among private variables.
    - Delete deprecated `neurom.features.register_neurite_feature`.

- Refactor morphology statistics, e.g. ``neurom stats`` command.
    - New config format. See :ref:`morph-stats-new-config`. The old format is still supported.
      The only necessary change is replace 'total' with 'sum', 'neuron' with 'morphology'.
    - Keep feature names as is. Don't trim 's' at the end of plurals.

- Delete ``neurom.check.structural_checks``, ``neurom.core.tree`` that were deprecated in v2.
- Delete unused ``neurom.utils.memoize``

Version 2.3.1
-------------
- fix ``features.neuronfunc._neuron_population`` for 'sholl_frequency' feature over a neuron
  population.
- use a tuple for ``subplot`` default value in ``view.common.get_figure``.

Version 2.3.0
-------------
- Introduce a new method to calculate partition asymmetry by Uylings.
  See docstring of `neurom.features.neuritefunc.partition_asymmetries`.
- Follow the same morphology validation rules as in MorphIO. See the :ref:`doc page<validation>`
  about it.
- Remove the cli command ``neurom features`` that listed all possible features. Instead a proper
  documentation is provided on that topic. See :func:`neurom.features.get`.
- Make ``neurom.features.neuronfunc.sholl_crossings`` private.
- Remove ``NeuriteType.all`` from ``NEURITES``

Version 2.2.1
-------------
- Fix 'section_path_lengths' feature for Population

Version 2.2.0
-------------
- Don't force loading of neurons into memory for Population (#922). See new API of
  :class:`Population<neurom.core.population.Population>` and `load_neurons<neurom.io.utils.load_neurons>`
- Move ``total_length`` feature to from ``neuritefunc`` to ``neuronfunc``. Use ``neurite_lengths``
  feature for neurites
- Include morphology filename extension into Neuron's name
- Extend ``tree_type_checker`` to accept a single tuple as an argument. Additionally validate
  function's arguments (#912, #914)
- Optimize Sholl analysis code (#905, #919)

Version 2.1.2
-------------
- Allow for morphologies without soma (#900)

Version 2.1.1
-------------
- Drop relative imports (keep backward compatibility) (#898)
- Account for all custom neurite types in NeuriteType (#902)
- Remove excessive pylint disables (#903)

Version 2.0.2
-------------
See a separate dedicated :ref:`page<migration-v2>` for it.

