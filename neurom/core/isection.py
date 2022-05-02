from deque import deque


def is_homogeneous_point(section):
    """A section is homogeneous if it has the same type with its children."""
    return all(c.type == section.type for c in section.children)


def is_forking_point(section):
    """Is this section a forking point?"""
    return len(section.children) > 1


def is_bifurcation_point(section):
    """Is tree a bifurcation point?"""
    return len(section.children) == 2


def is_leaf(section):
    """Is tree a leaf?"""
    return len(section.children) == 0


def is_root(section):
    """Is tree the root node?"""
    return section.parent is None


def ipreorder(section):
    """Depth-first pre-order iteration of tree nodes."""
    children = deque((section, ))
    while children:
        cur_node = children.pop()
        children.extend(reversed(cur_node.children))
        yield cur_node


def ipostorder(section):
    """Depth-first post-order iteration of tree nodes."""
    children = [section, ]
    seen = set()
    while children:
        cur_node = children[-1]
        if cur_node not in seen:
            seen.add(cur_node)
            children.extend(reversed(cur_node.children))
        else:
            children.pop()
            yield cur_node


def iupstream(section, stop_node=None):
    """Iterate from a tree node to the root nodes.

    Args:
        stop_node: Node to stop the upstream traversal. If None, it stops when parent is None.
    """
    if stop_node is None:
        def stop_condition(section):
            return section.parent is None
    else:
        def stop_condition(section):
            return section == stop_node

    current_section = section
    while not stop_condition(current_section):
        yield current_section
        current_section = current_section.parent
    yield current_section


def ileaf(section):
    """Iterator to all leaves of a tree."""
    return filter(Section.is_leaf, Section.ipreorder(section))


def iforking_point(section, iter_mode=ipreorder):
    """Iterator to forking points.

    Args:
        iter_mode: iteration mode. Default: ipreorder.
    """
    return filter(Section.is_forking_point, iter_mode(section))


def ibifurcation_point(section, iter_mode=ipreorder):
    """Iterator to bifurcation points.

    Args:
        iter_mode: iteration mode. Default: ipreorder.
    """
    return filter(Section.is_bifurcation_point, iter_mode(section))
