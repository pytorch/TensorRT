# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__name__), '../py'))

import sphinx_material
# -- Project information -----------------------------------------------------

project = 'TRTorch'
copyright = '2020, NVIDIA Corporation'
author = 'NVIDIA Corporation'

# The full version, including alpha/beta/rc tags
release = '0.0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'exhale',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_tmp', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Setup the breathe extension
breathe_projects = {
    "TRTorch": "./_tmp/xml"
}
breathe_default_project = "TRTorch"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./_cpp_api",
    "rootFileName":          "trtorch_cpp.rst",
    "rootFileTitle":         "TRTorch C++ API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../cpp/api/include"
}

html_show_sourcelink = True
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

extensions.append("sphinx_material")
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    'nav_title': 'TRTorch',
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://nvidia.github.io/TRTorch/',

    # Set the color and the accent color
    'theme_color': '84bd00',
    'color_primary': 'light-green',
    'color_accent': 'light-green',
    "html_minify": False,
    "html_prettify": True,
    "css_minify": True,
    "logo_icon": "&#xe86f",

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/nvidia/TRTorch/',
    'repo_name': 'TRTorch',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 2,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,
    'master_doc': True,
    "version_info": {
        "master": "https://nvidia.github.io/TRTorch/",
        "v0.0.2": "https://nvidia.github.io/TRTorch/v0.0.2/",
        "v0.0.1": "https://nvidia.github.io/TRTorch/v0.0.1/",
    }
}


# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'
cpp_id_attributes = ["TRTORCH_API"]


# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'