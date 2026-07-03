================
Causal Discovery
================

The Causal Discovery tool generates a directed acyclic graph (DAG) that represents the causal relationships between 
variables in your input dataset(s). This generated DAG can then serve as the foundational causal specification for 
your causal model.

.. note::
   Automated causal discovery is a starting point. The resulting DAG must always be manually inspected to ensure it 
   is a valid representation of your system.

Configuration Options
---------------------
The tool supports various configurations to tailor the discovery process to your specific dataset and domain knowledge:

* **Fitness Functions** Choose between score-based and multi-objective fitness functions.

* **Search Constraints (Iterations)**
  Adjust how long the algorithm runs, which is particularly useful for datasets with a large number of variables. 
  
  * *Default limits:* 100 maximum iterations, and 20 maximum iterations without improvement.

* **Domain Knowledge (Edge Constraints)**
  You can explicitly include or exclude specific edges in the output DAG using dot files. Regular expressions (regex) 
  are supported.

  **Example:** Consider the DAG for the `vaccinating the elderly 
  <https://github.com/CITCOM-project/CausalTestingFramework/tree/main/examples/covasim_/vaccinating_elderly>`_ 
  modelling scenario. This scenario has two inputs: ``vaccine`` and ``max_doses``.

  .. container:: zoom-container

     .. image:: ../../../examples/covasim_/vaccinating_elderly/dag.png
        :class: zoomable-image
        :alt: Causal DAG of the vaccinating the elderly modelling scenario

  * **Excluding Edges:** If domain knowledge dictates that ``max_doses`` has no causal effect on any outputs, you 
    can specify ``max_doses -> ".*"`` and ``".*" -> max_doses`` in the *exclude edges* dot file.
  * **Including Edges:** If it is known that ``vaccine`` directly affects all three outputs, you can specify 
    ``vaccine -> "cum_.*"`` in the *include edges* dot file.