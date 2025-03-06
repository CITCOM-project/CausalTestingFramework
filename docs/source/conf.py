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
from datetime import date

# Include root directory of project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# -- Project information -----------------------------------------------------

project = "Causal Testing Framework"
copyright = f"{date.today().year}, The CITCOM Team"
author = "Andrew Clark, Michael Foster, Neil Walkinshaw, Rob Hierons, Bob Turner, Christopher Wild, Farhad Allian"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["autoapi.extension", "myst_parser", "sphinx.ext.autosectionlabel"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', os.path.abspath('../../images')] # add /images directory to static path

html_css_files = ['css/custom.css']

# Path to generate documentation from using sphinx AutoAPI
autoapi_dirs = [os.path.abspath(os.path.join("..", "..", "causal_testing"))]

autoapi_generate_api_docs = True
autoapi_keep_files = True

# Suppress label warnings
suppress_warnings = ['autosectionlabel.*']


html_logo = '_static/images/CITCOM-logo.png'

html_theme_options = {
    'style_nav_header_background': '#9ADBE8',  # Set the colour using CSS
    'logo_only' : True,
}

