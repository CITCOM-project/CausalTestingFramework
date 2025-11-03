# Configuration file for the Sphinx documentation builder.
#
# For a full list of options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import date

# Include root directory of project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Allow Sphinx to find tutorial documents in /examples
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples")))


# -- Project information -----------------------------------------------------

project = "Causal Testing Framework"
copyright = f"{date.today().year}, the CITCoM team"
author = "Andrew Clark, Michael Foster, Neil Walkinshaw, Rob Hierons, Bob Turner, Christopher Wild, Farhad Allian"


# -- General configuration ---------------------------------------------------

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autosummary_imported_members = False

# Treat warnings as warnings, not errors
keep_warnings = True

# Don't fail on warnings
warning_is_error = False

# Suppress specific warning categories
suppress_warnings = [
    "app.add_directive",
    "app.add_role",
    "app.add_generic_role",
    "app.add_node",
    "autoapi.python_import_resolution",
    "autosectionlabel.*",
    "autodoc",
    "autodoc.import_object",
    "ref.python",
]

# Make Sphinx less strict about docstring parsing
nitpicky = False

# Configure nbsphinx
nbsphinx_execute = "never"

# Include /examples directory in output (for HTML build)
html_extra_path = [os.path.abspath("../../examples")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Patterns to exclude from the source directory
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Static files such as CSS or images
html_static_path = ["_static", os.path.abspath("../../images")]

# Custom CSS
html_css_files = ["css/custom.css"]

# Path to generate documentation from using Sphinx AutoAPI
autoapi_dirs = [os.path.abspath(os.path.join("..", "..", "causal_testing"))]
autoapi_generate_api_docs = True
autoapi_keep_files = False

# HTML logo and theme options
html_logo = "_static/images/CITCOM-logo.png"

html_theme_options = {
    "style_nav_header_background": "#9ADBE8",  # Set the colour using CSS
    "logo_only": True,
}
