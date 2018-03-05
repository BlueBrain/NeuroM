import numpy as np

import morphio
from neurom.core import Neurite, Neuron, Section, Soma
from neurom.core._soma import _SOMA_CONFIG, make_soma


def _section_builder(brain_section):
    points = np.concatenate((brain_section.points,
                             brain_section.diameters[:, np.newaxis] / 2.),
                            axis=1)

    section_id = brain_section.id-1
    section_type = brain_section.type
    section = Section(points, section_id, section_type)
    for child in brain_section.children:
        section.add_child(_section_builder(child ))
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

        if soma_points.size:
            soma = make_soma(soma_points , soma_class=soma_class)
        else:
            soma = Soma(points=np.empty((0,4)))
        super(BrionNeuron, self).__init__(soma=soma,
                                          name=name or 'Neuron',
                                          neurites=neurites)
