Background
=====================================


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

.. container:: zoom-container

   .. figure:: ../../images/schematic.png
      :class: zoomable-image
      :alt: Schematic diagram of the Causal Testing Framework
      :align: center

      **Figure:** Schematic diagram of the Causal Testing Framework.
      This figure illustrates the modular architecture and data flow between key components.

.. raw:: html

   <div style="margin-top: 30px;"></div>

The Causal Testing Framework primarily consists of the following 3 components: 1) Modelling Scenario, 2) Causal Graph, and 2) Causal Test Case.

#.
   :doc:`Causal Graph <../modules/causal_specification>`\ : To apply graphical causal inference techniques for testing, we need a *specification*.
   For this, we must specify the variables that are relevant to the modelling scenario of interest, and any constraints over them.
   We must also specify the expected causal relationships between the variables as a directed acyclic graph (DAG).


#.
   :doc:`Causal Tests <../modules/causal_testing>`\ : With a causal specification in hand, we can now design a series of test cases that interrogate the causal relationships of interest in the scenario-under-test.
   Informally, a causal test consists of an input configuration, an intervention which is applied to the input, and the expected *causal effect* of that intervention on some output.
   In other words, a causal test case states the expected causal effect of a particular intervention made to an input configuration.
   For each modelling scenario, the user should create a set of causal tests.
   Once a causal test case has been defined, it can be evaluated as follows:

   a. Using the causal specification, identify an estimand for the effect of the intervention on the output of interest, where the *estimand* represents a statistical procedure capable of estimating the causal effect of the intervention on the output.
   #. Apply a statistical estimator (e.g. ``linear regression``) to the data to obtain a point estimate for the causal effect.
      Depending on the estimator used, confidence intervals may also be obtained at a specified significance level, e.g. 0.05 corresponds to 95% confidence intervals (optional).
      The :doc:`Estimators Overview <../modules/estimators>` contains a list of the various estimators we support.
   #. Return the casual test result including a point estimate and 95% confidence intervals, usually quantifying the average treatment effect (ATE).
   #. Compare the estimated causal effect to the expected causal effect specified in the causal test case.
      The test passes if the two match, and fails otherwise.

For more information on each of these components, follow the links above to their respective module description pages.
