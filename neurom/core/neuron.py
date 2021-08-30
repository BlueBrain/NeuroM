"""For backward compatibility only."""
# pylint: skip-file

from neurom.core.morphology import *  # pragma: no cover
from neurom.utils import deprecated_module  # pragma: no cover

deprecated_module('Module `neurom.core.neuron` is deprecated. Use `neurom.core.morphology`'
                  ' instead.')  # pragma: no cover
