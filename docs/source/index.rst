.. Causal Testing documentation master file, created by

Welcome to Causal Testing's documentation!
==========================================

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises
graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box
perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect
relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to
design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect
of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the
system-under-test that is expected to cause a change to some output(s).

.. image:: /images/workflow.png

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   description
   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Module Descriptions

   /modules/data_collector
   /modules/causal_specification
   /modules/causal_tests

.. toctree::
   :maxdepth: 2
   :caption: API
   :titlesonly:

   /autoapi/causal_testing/index

.. toctree::
   :maxdepth: 1
   :caption: Front Ends

   frontends/json_front_end
   frontends/test_suite

.. toctree::
   :maxdepth: 1
   :caption: Glossary

   glossary

.. toctree::
   :maxdepth: 1
   :caption: Development Documentation

   /dev/version_release
   /dev/documentation
   /dev/actions_and_webhooks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`