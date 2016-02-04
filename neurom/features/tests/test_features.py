import numpy as np
import neurom.features as fs
from nose import tools as nt
from neurom.core.types import TreeType

def test_pkg():
    def f(): return (x for x in range(5))
    res = fs._pkg(f)()
    nt.assert_true(np.all(res == np.array([0.,1.,2.,3.,4.])))

def test_dispatch_feature_neurite():
    f1 = fs.NeuriteFeatures.segment_lengths
    rs = fs._dispatch_feature(f1)
    nt.assert_true(rs.__name__ == 'segment_lengths')

def test_dispatch_feature_neuron():
    f1 = fs.NeuronFeatures.soma_radius
    rs = fs._dispatch_feature(f1)
    nt.assert_true(rs.__name__ == 'soma_radius')

@nt.raises(TypeError)
def test_dispatch_feature_exception():

    fs._dispatch_feature(TreeType.axon)

def test_feature_factory():
    f1 = fs.NeuriteFeatures.segment_lengths
    rs = fs.feature_factory(f1)
    nt.assert_true(rs.__name__ == 'segment_lengths')