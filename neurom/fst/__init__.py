print('neurom.fst is being deprecated, please consider replacing it by neurom.features')

from neurom.features import neuritefunc as _neuritefunc
from neurom.features import neuronfunc as _neuronfunc
from neurom.features import sectionfunc

from neurom.features import NEURITEFEATURES, NEURONFEATURES, get, register_neurite_feature
