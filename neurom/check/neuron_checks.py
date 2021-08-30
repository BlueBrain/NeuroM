"""For backward compatibility only."""
# pylint: skip-file

from neurom.check.morphology_checks import *  # pragma: no cover
from neurom.utils import deprecated_module  # pragma: no cover

deprecated_module('Module `neurom.check.neuron_checks` is deprecated. Use'
                  '`neurom.check.morphology_checks` instead.')  # pragma: no cover
