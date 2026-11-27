Testing Causality in Scientific Modelling Software
==================================================

These tutorials reproduce the four case studies from Section 5 of our paper
`Testing Causality in Scientific Modelling Software <https://dl.acm.org/doi/10.1145/3607184>`_.
Each one walks through the causal testing workflow on a real scientific model, explaining every step along the way.
All the data you need is included with the Causal Testing Framework, so you don't need to install or run any of the
models yourself.

If you're new to causal testing, we recommend working through the tutorials in order.

:doc:`Poisson Line Tessellation <poisson_line_tessellation>` (Section 5.1)
   We test a simple stochastic model that scatters random lines over a window and counts the polygons they form.
   Starting with the original statistical metamorphic testing (SMT) study of this model, we show that the CTF reaches
   the same conclusions from 1000 general-purpose model runs that SMT needed 5000 purpose-built runs to reach.
   We then reuse the same data to test a second relation, revealing that the model's expected behaviour only holds for
   sufficiently large windows.

:doc:`Cardiac Action Potential <cardiac_action_potential>` (Section 5.2)
   We carry out a sensitivity analysis of the Luo-Rudy 1991 model of the electrical activity of a heart cell, testing
   how six ion channel conductances affect the duration of the action potential.
   Because the causal DAG shows that none of the effects are confounded, simple linear regression models are enough to
   reproduce the conclusions of an earlier study that used a much more complex Gaussian Process Emulator.

:doc:`Covasim: Prioritising Vaccination for the Elderly <covasim_vaccinating_elderly>` (Section 5.3)
   We test how prioritising the elderly for vaccination affects infections and vaccine uptake in Covasim, a widely
   used COVID-19 simulator.
   The data compares two groups of runs that differ only in their vaccination policy, making this a gentle introduction
   to causal testing with no confounding to worry about.
   Two of the four tests fail, uncovering a real bug in Covasim that was later fixed by its developers.

:doc:`Covasim: Testing with Confounded Data <covasim_disease_variants>` (Section 5.4)
   We predict how the transmissibility of different COVID-19 variants affects infections in each of the 156 locations
   Covasim can simulate, using observational data that is confounded by location.
   We use the causal DAG to find valid adjustment sets, choose between them using domain knowledge, and show that the
   CTF's estimates closely match a "gold standard" from 9360 SMT runs, using as few as around 200 runs.

.. toctree::
   :hidden:
   :maxdepth: 1

   poisson_line_tessellation
   cardiac_action_potential
   covasim_vaccinating_elderly
   covasim_disease_variants
