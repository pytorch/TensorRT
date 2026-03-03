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

import pytorch_sphinx_theme2
import torch
import torch_tensorrt
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

# -- Project information -----------------------------------------------------

project = "Torch-TensorRT"
copyright = "2024, NVIDIA Corporation"
author = "NVIDIA Corporation"

_raw_version = torch_tensorrt.__version__
version = f"v{_raw_version}"
# The full version, including alpha/beta/rc tags
release = f"v{_raw_version}"
# Dev/nightly builds include a git hash (e.g. v2.12.0.dev0+abc1234) which doesn't
# match any entry in versions.json — map those to "main" for the version switcher.
_version_match = "main" if ("dev" in _raw_version or "+" in _raw_version) else version

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
    "sphinx_gallery.gen_gallery",
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
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# Custom CSS paths should either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css",
    "css/custom.css",
]

# sphinx-gallery configuration
# Explicitly list which example subdirectories to render so we don't accidentally
# pick up legacy FX, raw triton client scripts, or training-only utilities.
sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/dynamo",
        "../examples/distributed_inference",
    ],
    "gallery_dirs": [
        "tutorials/_rendered_examples/dynamo",
        "tutorials/_rendered_examples/distributed_inference",
    ],
    # Exclude pure utility modules that aren't standalone runnable examples.
    "ignore_pattern": r"(utils\.py|rotary_embedding\.py|tensor_parallel_initialize_dist\.py)",
}

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

# Override the default "<page> — <project> <version> documentation" title,
# which produces the redundant "Torch-TensorRT — Torch-TensorRT vX documentation".
html_title = "Torch-TensorRT"

html_show_sourcelink = True
html_context = {
    "github_user": "pytorch",
    "github_repo": "TensorRT",
    "github_version": "main",
    "doc_path": "docsrc",
}
html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]
html_theme = "pytorch_sphinx_theme2"

html_theme_options = {
    "pytorch_project": "docs",
    "collapse_navigation": False,
    "display_version": True,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "version_switcher": {
        # Use the raw GitHub URL so the dropdown works in all environments
        # (local preview, staging, production) — pytorch.org only works once deployed.
        "json_url": "https://pytorch.org/TensorRT/versions.json",
        "version_match": _version_match,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pytorch/TensorRT/",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "show_version_warning_banner": False,
    "article_header_end": [],
    "article_footer_items": [],
    "footer_start": ["copyright"],
    "footer_end": [],
}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"
cpp_id_attributes = ["TORCHTRT_API"]

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

autodoc_type_aliases = {
    "LegacyConverterImplSignature": "LegacyConverterImplSignature",
    "DynamoConverterImplSignature": "DynamoConverterImplSignature",
    "ConverterImplSignature": "ConverterImplSignature",
}

nbsphinx_execute = "never"

autodoc_member_order = "groupwise"

# -- A patch that prevents Sphinx from cross-referencing ivar tags -------
# See http://stackoverflow.com/a/41184353/3343043

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docfields import TypedField


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
                        **kw,
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
