
Causal Specification
=====================

- In causal testing, our units of interest are specific usage **scenarios** of the system-under-test. For example,
  when testing an epidemiological computational model, one scenario could focus on the simulation of the spread of a virus in a crowded indoors space.
  For this scenario, our causal specification will describe how a number of interventions should **cause** some outputs to change e.g. opening a window should reduce the spread of the virus by some factor.

- In order to isolate the causal effect of the defined interventions, the user needs to express the anticipated cause-effect relationships amongst the inputs and outputs involved in the scenario.
  This is achieved using a causal DAG, a simple dot and arrow graph that does not contain any cycles where nodes are random variables that represent the
  inputs and outputs in the scenario-under-test, and edges represent causality. For example, ``window --> infection_prob`` encodes the belief that opening or closing the
  window should cause the probability of infection to change.

- A causal specification is simply the combination of these components: a series of requirements for the scenario-under-test and a causal DAG representing causality
  amongst the inputs and outputs.

- Collectively, the components of the causal specification provide both contextual information in the form of constraints and requirements, as well as causal information in the form of a causal DAG.
  It's these components that are used to design statistical experiments that can answer causal questions about the scenario-under-test.

1. Modelling Scenario
----------------------

- Each scenario is defined as a series of constraints placed over a set of input variables. A constraint is simply a mapping
  from an input variable to a specific value or distribution that characterises the scenario in question.
  For example, a scenario simulating the spread of a virus in a crowded indoors space would likely place a constraint over the size of room,
  the number of windows, and the number of people in the room.

- Requirements for this scenario should describe how a particular intervention
  (e.g. opening the window, changing the number of people, changing the size of the room etc.) is expected to cause a particular outcome (number of infections, deaths, R0 etc.) to change.
  The way these requirements are expressed is up to the user, however, it is essential that they focus on the expected effect of an intervention.

2. Causal DAG
--------------

In order to apply CI techniques, we need to capture causality amongst the inputs and outputs in the scenario-under-test.
Therefore, for each scenario, the user must define a causal DAG.
As an example, consider the DAG shown below for the `Poisson Line Process example. <https://github.com/CITCOM-project/CausalTestingFramework/tree/main/examples/poisson-line-process>`_
Here, the model has three inputs: ``width``, ``height``, and ``intensity``.
These inputs control the number of lines (``num_lines_abs``) and polygons (``num_shapes_abs``) that are drawn, which then feed into the numbers of lines (``num_lines_unit``) and polygons (``num_shapes_unit``) per unit area.
Note though that the ``num_lines_abs`` does not have a direct causal effect on ``num_shapes_unit``, since the number of polygons per unit area is defined entirely by the total number of polygons and the area of the sampling window.

.. container:: zoom-container

   .. image:: ../../../examples/poisson-line-process/dag.png
      :class: zoomable-image
      :alt: Schematic diagram of the Poisson Line Process DAG

.. literalinclude:: ../../../examples/poisson-line-process/dag.dot
   :language: graphviz
   :caption: **Figure:** Example Causal DAG for the Poisson line process example.

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
