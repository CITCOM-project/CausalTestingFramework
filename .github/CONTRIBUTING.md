# Contributing to Causal Testing Framework

### Questions
Ask any questions about the Causal Testing Framework or surrounding concepts on the
[discussions board](https://github.com/CITCOM-project/CausalTestingFramework/discussions). Before opening a new
discussion, please see whether a relevant one already exists - someone may have answered your question already.

### Reporting Bugs and Making Suggestions
Upon identifying any bugs or features that could be improved, please open an 
[issue](https://github.com/CITCOM-project/CausalTestingFramework/issues) and label with bug or suggestion. Every issue
should clearly explain the bug or feature to be improved and, where necessary, instructions to replicate.

### Making a Pull Request
In order to directly contribute to the Causal Testing Framework, the following steps must be taken:
1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the Causal Testing Framework repository.
    - While working on this fork, ensure that you 
      [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) 
      your main branch with `causal_testing_framework:main`.
2. Create a new [branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)
   from the latest main in your fork. 
   - This branch should have a name that describes the feature which is changed or added.
   - Work directly onto this branch, making sure that you follow our style guidelines outlined below.
3. Open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) 
   from your branch to the main branch.
   - Explain the changes made or feature added.
   - [Request a review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review).
   - The pull request will have to pass our [continuous integration (CI) checks](#continuous-integration-ci) and receive an 
     approving review, which will be determined by our [review guidelines]().
     
### Continuous Integration (CI)
Upon pushing or pulling, the following GitHub actions will be triggered:
    - Build: install Python dependencies
    - Linting: check style guidelines have been met.
    - Testing: run unit and regression tests.

### Coding Style
In the Causal Testing Framework, we aim to provide highly reusable and easily maintainable packages. To this end,
we ask contributors to stick to the following guidelines:
1. Make small and frequent changes rather than verbose and infrequent changes.
2. Favour readable and informative variable, method, and class names over concise names.
3. Use logging instead of print.
4. For every method and class, include detailed docstrings following the 
   [reStructuredText/Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) guidelines.
5. Add credit and license information where existing code has been used (in the method or class docstring).
6. If a method implements a specific algorithm/technique from a paper, add a citation to the docstring.
7. Use [variable](https://www.python.org/dev/peps/pep-0008/#variable-annotations) and [function](https://www.python.org/dev/peps/pep-0008/#function-annotations)
   annotations.
8. All methods should be thoroughly tested with PyTest (see [Testing]() below).
