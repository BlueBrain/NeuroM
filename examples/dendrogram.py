from neurom.view import common
from neurom.core.tree import isegment
from neurom.core.tree import ileaf
from neurom.core.tree import val_iter

from matplotlib.collections import LineCollection

from numpy import sqrt

def get_length(segment):

    p0, p1 = segment[0].value, segment[1].value

    # length calculated from the distance of the starting and
    # ending point in space 
    return sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)


def get_transformed_position(segment, segments_dict, y_length):

    start, end = segment

    start_node_id = start.value[-2]

    end_node_id = end.value[-2]

    #The parent node is connected with the child
    (x0, y0), visited = segments_dict[start_node_id]

    # increase horizontal dendrogram length by segment length
    x1 = x0 + get_length(segment)


    # if the parent node is visited for the first time
    # the dendrogram segment is drawn from below. If it
    # is visited twice it is drawn from above

    if visited == 0:

        y1 = y0 + y_length

        #     horizontal segment, vertical segment
        segments = [((x0, y1),(x1, y1)), ((x0, y0), (x0, y1))]

    elif visited == 1:

        y1 = y0 - y_length

        segments = [((x0,y1),(x1,y1)),((x0,y0),(x0,y1))]

    else:

        y1 = y0

        segments = [((x0,y1),(x1,y1))]


    if len(end.children) == 1:

        segments_dict[end_node_id] = [(x1, y1), -1]

    else:
        
        segments_dict[end_node_id] = [(x1, y1), 0]

    segments_dict[start_node_id][1] += 1

    return segments
    #return [horizontal_segment] + [vertical_segment]


def dendro_transform(tree_object):

    y_length = 2.

    segments_dict = {}

    segments_dict[tree_object.value[-2]] = [(0., 0.), 0.]

    positions = []
    diameters = []

    for seg in isegment(tree_object): 

        tr_pos = get_transformed_position(seg, segments_dict, y_length)

        seg_diam = seg[0].value[3] + seg[1].value[3]

        positions.extend(tr_pos)
        diameters.extend([seg_diam]*(len(tr_pos)/2))


    return positions, diameters




def dendrogram(tree_object, new_fig=True, subplot=False):
    '''docstring
    '''

    #number of terminations
    n_terminations = sum( 1 for _ in ileaf(tree_object))

    positions, linewidths = dendro_transform(tree_object)

    collection = LineCollection(positions, color='k',linewidth=linewidths)

    # Initialization of matplotlib figure and axes.
    fig, ax = common.get_figure(new_fig=new_fig, subplot=subplot)

    ax.add_collection(collection)
    ax.set_xlim([0.,500.])
    ax.set_ylim([-15.,15.])