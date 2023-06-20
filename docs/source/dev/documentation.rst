Project Documentation
=====================

This page aims to describe:

#. The projects documentation style

#. The tools used for documentation

#. ReadTheDocs where the documentation for this project is hosted


Documentation Style
-------------------

The `Sphinx docstring format <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format>`_ is used throughout the projects codebase to allow for the easy understanding of classes, methods, functions, etc.
This format allows for the easy generation of html documentation pages.

Checks for docstrings have been added to the projects Pylint configuration.

Documentation Tools
-------------------

To install the packages required to work with the documentation please ensure the projects **dev** dependencies are installed::

    pip install causal-testing-framework[dev]

Sphinx
******

This project makes use of `Sphinx <https://www.sphinx-doc.org/en/master/>`_, to generate documentation.

The documentation for the project sits within the `docs/` directory inside the project root.

To manually build the docs, first navigate to the `docs/` directory and run::

    make html

This will populate `docs/build/` with static html pages containing the docs.
To cleanup the compiled docs you can run::

    make clean


Situation within `docs/source` are the reStructuredText files (.rst) which contain the handwritten doc pages, which get compiled by the make commands.

Autodoc & Autoapi
*****************

`Autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ is an extension to sphinx that can import code modules and compile documentation from their docstrings.

`AutoAPI <https://sphinx-autoapi.readthedocs.io/en/latest/>`_ is a third party sphinx tool for recursively discovering code modules and compiling them into a logical doctree structure

The configuration for Sphinx, Autodoc and AutoAPI are all found in `/docs/source/conf.py <https://github.com/CITCOM-project/CausalTestingFramework/blob/main/docs/source/conf.py>`_.

ReadTheDocs
-----------
`Read the Docs <https://readthedocs.org/>`_ is a documentation hosting site that hosts, versions and builds documentation for free for open source projects.

This project makes use of a Github Webook to trigger the build in ReadTheDocs, further reading on this can be found :doc:`here <../dev/actions_and_webhooks>`\

