from nose import tools as nt

from neurom.core import features
from neurom.core.features import feature
from neurom.exceptions import NeuroMError


@feature(namespace='NEURONFEATURES')
def fake_feature():
    return None

def test_get_doc():
    nt.ok_('fake_feature' in features.get_doc())

def test_unknown_feature():
    nt.assert_raises(NeuroMError, features.get, 'feature_does_not_exist', None)
