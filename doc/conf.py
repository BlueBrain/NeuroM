# -*- coding: utf-8 -*-
#
# NeuroM documentation build configuration file, 
# created by on Tue April 28 2015.
#

import sys
import os
import sys, os
import neurom

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.1'

sys.path.append('sphinxext')

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.coverage',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'NeuroM'
copyright = u'HBP Algorithm Section'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 0.1
# The full version, including alpha/beta/rc tags.
release = 0.1

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

autosummary_generate = True
autodoc_default_flags = ['show-inheritance']
autoclass_content = 'both'
numpydoc_show_class_members = False

def _pelita_member_filter(parent_name, item_names):
    """
    Filter a list of autodoc items for which to generate documentation.

    Include only imports that come from the documented module or its
    submodules.

    """
    filtered_names = []

    if parent_name not in sys.modules:
        return item_names
    module = sys.modules[parent_name]

    for item_name in item_names:
        item = getattr(module, item_name, None)
        location = getattr(item, '__module__', None)

        if location is None or (location + ".").startswith(parent_name + "."):
            filtered_names.append(item_name)

    return filtered_names

# Using undocumented features of Jinja, not nice...
from jinja2.defaults import DEFAULT_NAMESPACE
DEFAULT_NAMESPACE['pelita_member_filter'] = _pelita_member_filter

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'haiku' # 'agogo'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/bbp.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'NeuroMDoc'

