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
  - name: Christopher Wild
    orcid: 0009-0009-1195-1497
    affiliation: 1
  - name: Farhad Allian
    affiliation: 1
  - name: Richard Somers
    orcid: 0009-0009-1195-1497
    affiliation: 1
  - name: Neil Walkinshaw
    orcid: 0000-0003-2134-6548
    affiliation: 1
  - name: Nicolas Lattimer
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
Metamorphic Testing is a popular technique for testing computational models (and other traditionally "hard to test" software).
Test goals are expressed as _metamorphic relations_ that specify how changing an input in a particular way should affect the software output.
Nondeterministic software can be tested using statistical metamorphic testing, which uses statistical tests over multiple executions of the software to determine whether the specified metamorphic relations hold.
However, this requires the software to be executed repeatedly for each set of parameters of interest, so is computationally expensive, and is constrained to testing properties over software inputs that can be directly and precisely controlled.
Statistical metamorphic testing cannot be used to test properties that relate internal variables or outputs to each other, since these cannot be controlled a priori.

By employing domain knowledge in the form of a causal graph --- a lightweight model specifying the expected relationships between key software variables --- the Causal Testing Framework circumvents both of these problems by enabling models to be tested using pre-existing runtime data.
The causal testing framework is written in python but is language agnostic in terms of the system under test.
All that is required is a set of properties to be validated, a causal model, and a set of software runtime data.

# Causal Testing
Causal Testing has four main steps, outlined in \ref{fig:schematic}.
Firstly, the user supplies a causal model, which takes the form of a directed acyclic graph (DAG) in which an edge $X \to Y$ represents variable $X$ having a direct causal effect on variable $Y$.
Secondly, the user supplies a set of causal properties to be tested.
Such properties can be generated from the causal DAG: for each $X \to Y$ edge, a test to validate the presence of a causal effect is generated, and for each missing edge, a test to validate independence is generated.
The user may also refine tests to validate the nature of a particular relationship.
Next, the user supplies a set of runtime data in the form of a table with each column representing a variable and rows containing the value of each variable for a particular run of the software.
Finally, the Causal Testing Framework automatically validates the supplied causal properties by using the supplied causal DAG and data to calculate a causal effect estimate, and validating this against the expected causal relationship.

![Causal Testing workflow.\label{fig:schematic}](images/schematic.png)

## Test Adequacy
Because the properties being tested are completely separate from the data used to validate them, traditional coverage-based metrics are not appropriate here.
The Causal Testing Framework instead evaluates the adequacy of a particular dataset by calculating a statistical metric based on the stability of the causal effect estimate, with numbers closer to zero representing more adequate data.

## Missing Variables
Causal Testing works by using the supplied causal DAG to identify those variables which need to be statistically controlled for to remove their biassing effect on the causal estimate.
This typically means we need to know their values.
However, the Causal Testing Framework can still sometimes estimate unbiased causal effects using Instrumental Variables, an advanced Causal Inference technique.

## Feedback
Many scientific models involve iterating several interacting processes over time.
These processes often feed into each other, and can create feedback cycles.
Traditional Causal Inference cannot handle this, however the Causal Testing Framework uses another advanced Causal Inference technique, g-methods, to enable the estimation of causal effects even when there are feedback cycles between variables.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

Example paper.bib file:

@article{Pearson:2017,
  	url = {http://adsabs.harvard.edu/abs/2017arXiv170304627P},
  	Archiveprefix = {arXiv},
  	Author = {{Pearson}, S. and {Price-Whelan}, A.~M. and {Johnston}, K.~V.},
  	Eprint = {1703.04627},
  	Journal = {ArXiv e-prints},
  	Keywords = {Astrophysics - Astrophysics of Galaxies},
  	Month = mar,
  	Title = {{Gaps in Globular Cluster Streams: Pal 5 and the Galactic Bar}},
  	Year = 2017
}

@book{Binney:2008,
  	url = {http://adsabs.harvard.edu/abs/2008gady.book.....B},
  	Author = {{Binney}, J. and {Tremaine}, S.},
  	Booktitle = {Galactic Dynamics: Second Edition, by James Binney and Scott Tremaine.~ISBN 978-0-691-13026-2 (HB).~Published by Princeton University Press, Princeton, NJ USA, 2008.},
  	Publisher = {Princeton University Press},
  	Title = {{Galactic Dynamics: Second Edition}},
  	Year = 2008
}

@article{gaia,
    author = {{Gaia Collaboration}},
    title = "{The Gaia mission}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1609.04153},
    primaryClass = "astro-ph.IM",
    keywords = {space vehicles: instruments, Galaxy: structure, astrometry, parallaxes, proper motions, telescopes},
    year = 2016,
    month = nov,
    volume = 595,
    doi = {10.1051/0004-6361/201629272},
    url = {http://adsabs.harvard.edu/abs/2016A%26A...595A...1G},
}

@article{astropy,
    author = {{Astropy Collaboration}},
    title = "{Astropy: A community Python package for astronomy}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1307.6212},
    primaryClass = "astro-ph.IM",
    keywords = {methods: data analysis, methods: miscellaneous, virtual observatory tools},
    year = 2013,
    month = oct,
    volume = 558,
    doi = {10.1051/0004-6361/201322068},
    url = {http://adsabs.harvard.edu/abs/2013A%26A...558A..33A}
}

@misc{fidgit,
  author = {A. M. Smith and K. Thaney and M. Hahnel},
  title = {Fidgit: An ungodly union of GitHub and Figshare},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/arfon/fidgit}
}
