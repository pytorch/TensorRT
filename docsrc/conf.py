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

sys.path.append(os.path.join(os.path.dirname(__name__), "../py"))

import torch
import pytorch_sphinx_theme
import torch_tensorrt

# -- Project information -----------------------------------------------------

project = "Torch-TensorRT"
copyright = "2021, NVIDIA Corporation"
author = "NVIDIA Corporation"

version = "master (" + torch_tensorrt.__version__ + ")"
# The full version, including alpha/beta/rc tags
release = "master"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "exhale",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

napoleon_use_ivar = True

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_tmp", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Setup the breathe extension
breathe_projects = {"Torch-TensorRT": "./_tmp/xml"}
breathe_default_project = "Torch-TensorRT"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./_cpp_api",
    "rootFileName": "torch_tensort_cpp.rst",
    "rootFileTitle": "Torch-TensorRT C++ API",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../cpp/include",
}

html_show_sourcelink = True
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# extensions.append("sphinx_material")
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
# html_context = sphinx_material.get_html_context()
html_theme = "pytorch_sphinx_theme"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "Torch-TensorRT",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://nvidia.github.io/Torch-TensorRT/",
    # Set the color and the accent color
    "theme_color": "84bd00",
    "color_primary": "light-green",
    "color_accent": "light-green",
    "html_minify": False,
    "html_prettify": True,
    "css_minify": True,
    "logo_icon": "&#xe86f",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/pytorch/TensorRT/",
    "repo_name": "Torch-TensorRT",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 1,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": True,
    "master_doc": True,
    "version_info": {
        "master": "https://nvidia.github.io/Torch-TensorRT/",
        "v1.1.0": "https://nvidia.github.io/Torch-TensorRT/v1.1.0/",
        "v1.0.0": "https://nvidia.github.io/Torch-TensorRT/v1.0.0/",
        "v0.4.1": "https://nvidia.github.io/Torch-TensorRT/v0.4.1/",
        "v0.4.0": "https://nvidia.github.io/Torch-TensorRT/v0.4.0/",
        "v0.3.0": "https://nvidia.github.io/Torch-TensorRT/v0.3.0/",
        "v0.2.0": "https://nvidia.github.io/Torch-TensorRT/v0.2.0/",
        "v0.1.0": "https://nvidia.github.io/Torch-TensorRT/v0.1.0/",
        "v0.0.3": "https://nvidia.github.io/Torch-TensorRT/v0.0.3/",
        "v0.0.2": "https://nvidia.github.io/Torch-TensorRT/v0.0.2/",
        "v0.0.1": "https://nvidia.github.io/Torch-TensorRT/v0.0.1/",
    },
}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"
cpp_id_attributes = ["TORCHTRT_API"]

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

# -- A patch that prevents Sphinx from cross-referencing ivar tags -------
# See http://stackoverflow.com/a/41184353/3343043

from docutils import nodes
from sphinx.util.docfields import TypedField
from sphinx import addnodes


def patched_make_field(self, types, domain, items, **kw):
    # `kw` catches `env=None` needed for newer sphinx while maintaining
    #  backwards compatibility when passed along further down!

    # type: (list, unicode, tuple) -> nodes.field
    def handle_item(fieldarg, content):
        par = nodes.paragraph()
        par += addnodes.literal_strong("", fieldarg)  # Patch: this line added
        # par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
        #                           addnodes.literal_strong))
        if fieldarg in types:
            par += nodes.Text(" (")
            # NOTE: using .pop() here to prevent a single type node to be
            # inserted twice into the doctree, which leads to
            # inconsistencies later when references are resolved
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = "".join(n.astext() for n in fieldtype)
                typename = typename.replace("int", "python:int")
                typename = typename.replace("long", "python:long")
                typename = typename.replace("float", "python:float")
                typename = typename.replace("type", "python:type")
                par.extend(
                    self.make_xrefs(
                        self.typerolename,
                        domain,
                        typename,
                        addnodes.literal_emphasis,
                        **kw
                    )
                )
            else:
                par += fieldtype
            par += nodes.Text(")")
        par += nodes.Text(" -- ")
        par += content
        return par

    fieldname = nodes.field_name("", self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item("", handle_item(fieldarg, content))
    fieldbody = nodes.field_body("", bodynode)
    return nodes.field("", fieldname, fieldbody)


TypedField.make_field = patched_make_field
