
Causal Specification
=====================

As in traditional software testing, the specification defines the expected behaviour of the system.
In causal testing, this is made up of two components: the modelling scenario and the causal graph.
These components are then used to design statistical experiments that can answer causal questions about the system-under-test.

1. Modelling Scenario
---------------------

- In causal testing, our units of interest are specific usage **scenarios** of the system-under-test.
  For example, when testing an epidemiological computational model, one scenario could focus on the simulation of the spread of a virus through a population.
  For this scenario, we may then test how a number of interventions should **cause** some outputs to change e.g. vaccinations should reduce the total number of deaths.

- Each scenario is defined as a series of constraints placed over a set of input variables.
  A constraint is simply a mapping from an input variable to a specific value or distribution that characterises the scenario in question.
  For example, a scenario simulating the spread of a virus would likely place constraints on the location, population demographics, and who is vaccinated.

- Requirements for this scenario should describe how a particular intervention
  (e.g.changing the number of people, changing who is vaccinated, etc.) is expected to cause a particular outcome (number of infections, deaths, R0, etc.) to change.
  The way these requirements are expressed is up to the user, however, it is essential that they focus on the expected effect of an intervention.

2. Causal Graph
---------------

To isolate the causal effect of the defined interventions, the user needs to express the anticipated cause-effect relationships amongst the inputs and outputs involved in the scenario.
This is done using a directed acyclic graph (DAG) in which nodes represent variables in the system and edges represent causal effects.
In order to apply CI techniques, we need to capture causality amongst the inputs and outputs in the scenario-under-test.
Therefore, for each scenario, the user must define a causal DAG.

As an example, consider the DAG shown below for the `vaccinating the elderly example. <https://github.com/CITCOM-project/CausalTestingFramework/tree/main/examples/covasim_/vaccinating_elderly>`_
This modelling scenario has two inputs `vaccine` and `max_doses` and three outputs `cum_vaccinations`, `cum_vaccinated`, and `cum_infections`.
We do not expect `max_doses` to have a causal effect on any of the outputs since this remains constant throughout modelling scenario.

.. container:: zoom-container

   .. image:: ../../../examples/covasim_/vaccinating_elderly/dag.png
      :class: zoomable-image
      :alt: Causal DAG of the vaccinating the elderly modelling scenario

.. literalinclude:: ../../../examples/covasim_/vaccinating_elderly/dag.dot
   :language: graphviz
   :caption: **Figure:** Example Causal DAG for the vaccinating the elderly example.

.. raw:: html

   <style>
   .zoom-overlay {
       position: fixed;
       top: 0;
       left: 0;
       width: 100%;
       height: 100%;
       background-color: rgba(0, 0, 0, 0.7);
       display: flex;
       align-items: center;
       justify-content: center;
       z-index: 9999;
   }

   .zoom-container {
       cursor: zoom-in;
       transition: transform 1s ease-in-out;
       background-color: white;
   }

   .zoom-container.zoomed {
       transform: scale(4);
       cursor: zoom-out;
   }

   .zoomable-image {
       max-width: 100%;
       max-height: 100%;
       margin: auto;
       background-color: white;
   }

   .zoom-container:hover {
       cursor: zoom-in;
   }
   </style>

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', function () {
       var image = document.querySelector('.zoomable-image');

       image.addEventListener('click', function () {
           var overlay = document.createElement('div');
           overlay.className = 'zoom-overlay';

           var container = document.createElement('div');
           container.className = 'zoom-container';
           container.style.cursor = 'zoom-out';

           var clonedImage = image.cloneNode();
           clonedImage.classList.add('zoomable-image');
           container.appendChild(clonedImage);
           overlay.appendChild(container);

           document.body.appendChild(overlay);

           overlay.addEventListener('click', function () {
               overlay.remove();
           });
       });
   });
   </script>

.. note::

   Unfortunately, there is no universally applicable guidance or algorithm that can be followed to create a causal DAG, but there are three general requirements that should be satisfied:

   1. The DAG must contain all inputs and outputs involved in the scenario-under-test.

   2. If there are any other variables which are not directly involved but are expected to have a causal relationship with the variables in the scenario-under-test, these should also be added to the graph. For example, the size of the room might be partially caused by the simulated location (house styles, average wealth, etc.), in which case location should be added to the DAG with an edge to room size and any other variables it is deemed to influence.

   3. If in doubt, add an edge. It’s a stronger assumption to exclude an edge (X and Y are independent) than to include one (X has some potentially negligible causal effect on Y).
