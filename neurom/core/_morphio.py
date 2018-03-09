'''Compatibility layer to create NeuroM neurons from MorphIO objects'''
import numpy as np

from neurom import Morphology
from neurom.core import Neurite, Neuron, Section, Soma
from neurom.core._soma import make_soma


def _section_builder(brain_section):
    '''Build a NeuroM section from a MorphIO one'''
    points = np.concatenate((brain_section.points,
                             brain_section.diameters[:, np.newaxis] / 2.),
                            axis=1)

    section_id = brain_section.id - 1
    section_type = brain_section.type
    section = Section(points, section_id, section_type)
    for child in brain_section.children:
        section.add_child(_section_builder(child))
    return section


class MorphioNeuron(Neuron):
    '''The MorphIO neuron wrapper'''

    def __init__(self, handle, name=None):
        '''Create a MorphIO neuron'''
        morphology = Morphology(handle)
        neurites = [Neurite(_section_builder(root_node))
                    for root_node in morphology.rootSections]

        brain_soma = morphology.soma
        soma_points = np.concatenate((brain_soma.points,
                                      brain_soma.diameters[:, np.newaxis] / 2.),
                                     axis=1)

        if soma_points.size:
            soma = make_soma(morphology.somaType, soma_points)
        else:
            soma = Soma(points=np.empty((0, 4)))
        super(MorphioNeuron, self).__init__(soma=soma,
                                            name=name or 'Neuron',
                                            neurites=neurites)
