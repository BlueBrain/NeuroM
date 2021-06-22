Changelog
=========

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

