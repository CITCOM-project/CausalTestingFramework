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
  - name: Nicholas Lattimer
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
In fields such as epidemiology, where researchers seek answers to challenging causal questions, a statistical methodology known as Causal Inference has addressed similar problems, enabling the inference of causal conclusions from noisy, biased, and sparse observational data instead of costly randomised trials.
Causal Inference works by using domain knowledge to identify and mitigate for biases in the data, enabling them to answer causal questions that concern the effect of changing some feature on the observed outcome.
The Causal Testing Framework is a software testing framework that uses Causal Inference techniques to establish causal effects between software variables from pre-existing runtime data rather than having to collect bespoke, highly curated datasets especially for testing.

# Statement of need
Metamorphic Testing @[chen1998metamorphic] is a popular technique for testing computational models (and other traditionally "hard to test" software).
Test goals are expressed as _metamorphic relations_ that specify how changing an input in a particular way should affect the software output.
Nondeterministic software can be tested using Statistical Metamorphic Testing @[guderlei2007smt], which uses statistical tests over multiple executions of the software to determine whether the specified metamorphic relations hold.
However, this requires the software to be executed repeatedly for each set of parameters of interest, so is computationally expensive, and is constrained to testing properties over software inputs that can be directly and precisely controlled.
Statistical Metamorphic Testing cannot be used to test properties that relate internal variables or outputs to each other, since these cannot be controlled a priori.

By employing domain knowledge in the form of a causal graph --- a lightweight model specifying the expected relationships between key software variables --- the Causal Testing Framework circumvents both of these problems by enabling models to be tested using pre-existing runtime data.
The Causal Testing Framework is written in python but is language agnostic in terms of the system under test.
All that is required is a set of properties to be validated, a causal model, and a set of software runtime data.

# Causal Testing
Causal Testing @[clark2023testing] has four main steps, outlined in \ref{fig:schematic}.
Firstly, the user supplies a causal model, which takes the form of a directed acyclic graph (DAG) in which an edge $X \to Y$ represents variable $X$ having a direct causal effect on variable $Y$.
Secondly, the user supplies a set of causal properties to be tested.
Such properties can be generated from the causal DAG @[clark2023metamorphic]: for each $X \to Y$ edge, a test to validate the presence of a causal effect is generated, and for each missing edge, a test to validate independence is generated.
The user may also refine tests to validate the nature of a particular relationship.
Next, the user supplies a set of runtime data in the form of a table with each column representing a variable and rows containing the value of each variable for a particular run of the software.
Finally, the Causal Testing Framework automatically validates the supplied causal properties by using the supplied causal DAG and data to calculate a causal effect estimate, and validating this against the expected causal relationship.

![Causal Testing workflow.\label{fig:schematic}](images/schematic.png)

## Test Adequacy
Because the properties being tested are completely separate from the data used to validate them, traditional coverage-based metrics are not appropriate here.
The Causal Testing Framework instead evaluates the adequacy of a particular dataset by calculating a statistical metric @[foster2024adequacy] based on the stability of the causal effect estimate, with numbers closer to zero representing more adequate data.

## Missing Variables
Causal Testing works by using the supplied causal DAG to identify those variables which need to be statistically controlled for to remove their biassing effect on the causal estimate.
This typically means we need to know their values.
However, the Causal Testing Framework can still sometimes estimate unbiased causal effects using Instrumental Variables, an advanced Causal Inference technique.

## Feedback Over Time
Many scientific models involve iterating several interacting processes over time.
These processes often feed into each other, and can create feedback cycles.
Traditional Causal Inference cannot handle this, however the Causal Testing Framework uses another advanced Causal Inference technique, g-methods, to enable the estimation of causal effects even when there are feedback cycles between variables.

# Acknowledgements
This work was supported by the EPSRC CITCoM grant EP/T030526/1.
