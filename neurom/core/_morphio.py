import numpy as np

import morphio
from neurom.core import Neurite, Neuron, Section
from neurom.core._soma import _SOMA_CONFIG, make_soma

def _section_builder(brain_section, is_root=True):
    points = np.concatenate((brain_section.points,
                             brain_section.diameters[:, np.newaxis] / 2.),
                            axis=1)

    section_id = brain_section.id
    section_type = brain_section.type
    section = Section(points, section_id, section_type)
    for child in brain_section.children:
        section.add_child(_section_builder(child, is_root=False))
    return section


class BrionNeuron(Neuron):
    def __init__(self, handle, name=None):
        morphology = morphio.Morphology(handle)
        neurites = [Neurite(_section_builder(root_node))
                    for root_node in morphology.rootSections]

        brain_soma = morphology.soma
        soma_points = np.concatenate((brain_soma.points,
                                 brain_soma.diameters[:, np.newaxis] / 2.),
                                axis=1)


        soma_check, soma_class = _SOMA_CONFIG.get(morphology.version)

        soma = make_soma(soma_points , soma_class=soma_class)

        super(BrionNeuron, self).__init__(soma=soma,
                                          name=name or 'Neuron',
                                          neurites=neurites)
