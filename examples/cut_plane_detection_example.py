'''Cut plane detection examples'''
import logging


from neurom import load_neuron
from neurom.apps.cut_plane_detection import find_cut_plane

L = logging.getLogger(__name__)


def demo():
    '''Examples of how to use find_cut_plane'''
    neuron_slice = load_neuron('../test_data/valid_set/Neuron_slice.h5')

    result = find_cut_plane(neuron_slice, bin_width=1)
    print('Cut plane found using a bin width of 1 (which is too low): {0}' .format(
        result['cut_plane']))
    print('Status: {0}'.format(result['status']))

    result = find_cut_plane(neuron_slice)
    print('\nCut plane found using default parameter: {0}'.format(
        result['cut_plane']))
    print('Status: {0}'.format(result['status']))
    print('Number of leaves to repair: {0}'.format(len(result['cut_leaves'])))
    print('More details can be found under details:\n{0}'.format(result['details']))

    try:
        print('\nNow displaying the plots using the display=True option...')
        import matplotlib.pyplot as plt
        result = find_cut_plane(neuron_slice, display=True)
        for name, (fig, _) in result['figures'].items():
            print('Saving figure {}.png'.format(name))
            fig.savefig(name)
        print("\nRemember:"
              " it's up to the user to call matplotlib.pyplot.show() to display the plots")
        plt.show()
    except ImportError:
        L.warning(
            "It appears that matplotlib is not installed. Can't display plots.")


demo()
