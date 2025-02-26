---
title: 'The Causal Testing Framework'
tags:
  - Python
  - causal testing
  - causal inference
  - causality
  - software testing
  - metamorphic testing
authors:
  - name: Michael Foster
    orcid: 0000-0001-8233-9873
    affiliation: 1
    corresponding: true
  - name: Andrew Clark
    orcid: 0000-0002-6830-0566
    affiliation: 1
  - name: Christopher Wild
    orcid: 0009-0009-1195-1497
    affiliation: 1
  - name: Farhad Allian
    orcid: 0000-0002-4569-0370
    affiliation: 1
  - name: Robert Turner
    orcid: 0000-0002-1353-1404
    affiliation: 1
  - name: Richard Somers
    orcid: 0009-0009-1195-1497
    affiliation: 1
  - name: Nicholas Latimer
    orcid: 0000-0001-5304-5585
    affiliation: 1
  - name: Neil Walkinshaw
    orcid: 0000-0003-2134-6548
    affiliation: 1
  - name: Rob Hierons
    orcid: 0000-0003-2134-6548
    affiliation: 1
affiliations:
 - name: University of Sheffield, UK
   index: 1
date: 2 December 2024
bibliography: paper.bib
---

# Summary
Scientific models possess several properties that make them notoriously difficult to test, including a complex input space, long execution times, and non-determinism, rendering existing testing techniques impractical.
In fields such as epidemiology, where researchers seek answers to challenging causal questions, a statistical methodology known as Causal Inference (CI) [@pearl2009causality; @hernan2020causal] has addressed similar problems, enabling the inference of causal conclusions from noisy, biased, and sparse observational data instead of costly randomised trials.
CI works by using domain knowledge to identify and mitigate for biases in the data, enabling them to answer causal questions that concern the effect of changing some feature on the observed outcome.
The Causal Testing Framework (CTF) is a software testing framework that uses CI techniques to establish causal effects between software variables from pre-existing runtime data rather than having to collect bespoke, highly curated datasets especially for testing.

# Statement of need
Metamorphic Testing [@chen1998metamorphic] is a popular technique for testing computational models (and other traditionally "hard to test" software).
Test goals are expressed as _metamorphic relations_ that specify how changing an input in a particular way should affect the software output.
Nondeterministic software can be tested using Statistical Metamorphic Testing [@guderlei2007smt], which uses statistical tests over multiple executions of the software to determine whether the specified metamorphic relations hold.
However, this requires the software to be executed repeatedly for each set of parameters of interest, so is computationally expensive, and is constrained to testing properties over software inputs that can be directly and precisely controlled.
Statistical Metamorphic Testing cannot be used to test properties that relate internal variables or outputs to each other, since these cannot be controlled a priori.

By employing domain knowledge in the form of a causal graph---a lightweight model specifying the expected relationships between key software variables---the CTF overcomes the limitations of Statistical Metamorphic Testing by enabling models to be tested using pre-existing runtime data.
The CTF is written in Python but is language agnostic in terms of the system under test.
All that is required is a set of properties to be validated, a causal model, and a set of software runtime data.

# Causal Testing
Causal Testing [@clark2023testing] has four main steps, outlined in Figure \ref{fig:schematic}.
Firstly, the user supplies a causal model, which takes the form of a directed acyclic graph (DAG) [@pearl2009causality] where an edge $X \to Y$ represents variable $X$ having a direct causal effect on variable $Y$.
Secondly, the user supplies a set of causal properties to be tested.
Such properties can be generated from the causal DAG [@clark2023metamorphic]: for each $X \to Y$ edge, a test to validate the presence of a causal effect is generated, and for each missing edge, a test to validate independence is generated.
The user may also refine tests to validate the nature of a particular relationship.
Next, the user supplies a set of runtime data in the form of a table with each column representing a variable and rows containing the value of each variable for a particular run of the software.
Finally, the CTF automatically validates the causal properties by using the causal DAG to identify a statistical estimand [@pearl2009causality] (essentially a set of features in the data which must be controlled for), calculate a causal effect estimate from the supplied data, and validating this against the expected causal relationship.

![Causal Testing workflow.\label{fig:schematic}](../images/schematic.png)

## Test Adequacy
Because the properties being tested are completely separate from the data used to validate them, traditional coverage-based metrics are not appropriate here.
The CTF instead evaluates the adequacy of a particular dataset by calculating a statistical metric [@foster2024adequacy] based on the stability of the causal effect estimate, with numbers closer to zero representing more adequate data.

## Missing Variables
Causal Testing works by using the causal DAG to identify the variables that need to be statistically controlled for to remove their biassing effect on the causal estimate.
This typically means we need to know their values.
However, where such biassing variables are not recorded in the data, the Causal Testing Framework can still sometimes estimate unbiased causal effects by using Instrumental Variables [@hernan2020causal], an advanced Causal Inference technique.

## Feedback Over Time
Many scientific models involve iterating several interacting processes over time.
These processes often feed into each other, and can create feedback cycles.
Traditional CI cannot handle this, however the CTF uses a family of advanced CI techniques, called g-methods [@hernan2020causal], to enable the estimation of causal effects even when there are feedback cycles between variables.

# Related Work
The Dagitty tool [@textor2017dagitty] is a browser-based environment for creating, editing, and analysing causal graphs.
There is also an R package for local use, but Dagitty cannot be used to estimate causal effects.
For this, doWhy [@sharma2020dowhy; @blobaum2024dowhy] is a free, open source Python package, and [cStruture](https://cstructure.dev) is a paid low code CI platform.
However, these packages are intended for general CI.
Neither explicitly supports causal software testing, nor do they support temporal feedback loops.

# Ongoing and Future Research
The CTF is the subject of several publications [@clark2023metamorphic; @clark2023testing; @foster2024adequacy; @somers2024configuration].
We are also in the process of preparing scientific publications concerning how the CTF handles missing variables and feedback over time.
Furthermore, we are working to develop a plug-in for the [DAFNI platform](https://www.dafni.ac.uk/) to enable national-scale infrastructure models to be easily tested.

# Acknowledgements
This work was supported by the EPSRC CITCoM grant EP/T030526/1.

# References
