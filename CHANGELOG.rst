Changelog
=========

Version 2.4.0
-------------
- Refactor plotting functionality. :ref:`migration-v2.4.0`.
    - deprecate ``neurom.view.viewer``
    - swap arguments ``ax`` and ``nrn`` of all plot functions in ``neurom.view.view``
    - delete ``neurom.view.plotly.draw``. Use instead ``neurom.view.plotly.plot_neuron`` and
      ``neurom.view.plotly.plot_neuron3d``.
    - rename ``neurom.view.view`` to ``neurom.view.matplotlib_impl``
    - rename ``neurom.view.plotly`` to ``neurom.view.plotly_impl``
    - rename ``neurom.view.common`` to ``neurom.view.matplotlib_utils``

- Refactor features.
    - Rigid classification of features. ``neuritefunc`` features must accept only a single neurite.
      ``neuronfunc`` features must accept only a single neuron. ``populationfunc`` features must
      accept only a collection of neurons or a neuron population.
    - Some features were deleted, renamed, added. See :ref:`migration-v2.4.0`.
    - Name consistency among private variables.
    - Delete deprecated `neurom.features.register_neurite_feature`.

- Refactor morphology statistics, e.g. ``neurom stats`` command.
    - New config format. See :ref:`morph-stats-new-config`. The old format is deprecated.
    - Use `sum` instead of `total` mode in config
    - Keep feature names as is. Don't trim 's' at the end of plurals.

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

