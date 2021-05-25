Changelog
=========

Version 2.2.0
-------------
- Move ``total_length`` feature to from ``neuritefunc`` to ``neuronfunc``. Use ``neurite_lengths``
  feature for neurites
- Include morphology filename extension into Neuron's name
- Don't force loading of neurons into memory for Population (#922). API of Population
  and `load_neurons` changed, see their code documentation.
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

