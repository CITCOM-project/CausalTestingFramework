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

* **Technique:** 
  Choose the causal discovery algorithm to use. Currently supported techniques are HillClimb and 
  NSGA. You can add custom techniques by implementing abstract_discovery.py and adding the new technique to your 
  endpoints in pyproject.toml.

  For additional control on the search process, you can specify the following optional parameters: 

  * For both techniques, you can specify the ``max_iterations`` and ``random_seed`` parameters. As well as domain 
    knowledge constraints (see below).

  * For the HillClimbing technique, you can specify the ``max_iterations_without_improvement`` parameter. Where 
    max_iterations_without_improvement is the number of iterations after an improvement is found, before the algorithm 
    widens its search space to avoid getting stuck in a local minima. 
  
  * If you are using the NSGA technique, you can specify the ``population_size`` and ``num_parents_mating`` parameters.

  Parameter defaults:
    | ``max_iterations``: 100
    | ``max_iterations_without_improvement``: 10
    | ``population_size``: 5
    | ``num_parents_mating``: 2
    | ``random_seed``: 0

* **Domain Knowledge (Edge Constraints)** You can explicitly include or exclude specific edges in the output DAG using 
  dot files. Regular expressions (regex) are supported.

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
  * **Specifying Variables:** You can specify the variables to include in the discovery process using the 
    ``--variables`` argument. If not specified, all variables in the input dataset(s) will be considered.
* **Examples:** To generate a DAG using the HillClimb technique with a maximum of 500 iterations, with 25 iterations 
  without improvement, a random seed of 63, you can use the following command:

  .. code-block:: bash
    
    causal-testing discover \
        --technique HillClimberDiscovery \
        --data-paths /test_data1.csv /test_data2.csv \
        --output /tmp/resultant_dag.dot \
        --technique-kwargs max_iterations=500 max_iterations_without_improvement=25 \ 
        random_seed=63

  Or to generate a DAG using the NSGA technique with a population size of 10, a num_parents_mating of 3, and 
  specified included and excluded edges, you can use the following command:

  .. code-block:: bash
    
    causal-testing discover \
        --technique NSGADiscovery \
        --data-paths /test_data1.csv \
        --output /tmp/resultant_dag.dot \
        --include-edges /include_edges.dot \
        --exclude-edges /exclude_edges.dot \
        --technique-kwargs population_size=10 num_parents_mating=3 