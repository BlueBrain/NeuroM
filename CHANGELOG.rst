Changelog
=========

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
- Introduce a new method to calculate partition asymmetry by Uylings. See docstring of
  :func:`neurom.features.neuritefunc.partition_asymmetries`.
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
  :class:`Population<neurom.core.population.Population>` and
  :func:`load_neurons<neurom.io.utils.load_neurons>`
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

