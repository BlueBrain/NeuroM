=======================
Migration to v2 version
=======================

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
    - no soma
    - 2 point soma
    - unknown soma type
    - disconnected points
    - non-consequent ids
    The following is an invalid morphology:
    - custom neurite type (5 or more)
