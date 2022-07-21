.. Causal Testing documentation master file, created by
   sphinx-quickstart on Thu Nov 11 12:31:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Causal Testing's documentation!
==========================================

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the system-under-test that is expected to cause a change to some output(s).

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   installation

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :titlesonly:

   /autoapi/causal_testing/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   json_frontend

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`