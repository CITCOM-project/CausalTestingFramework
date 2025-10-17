# How to Contribute

### Questions
Please ask any questions about the Causal Testing Framework or surrounding concepts on the
[discussions board](https://github.com/CITCOM-project/CausalTestingFramework/discussions). Before opening a new
discussion, please see whether a relevant one already exists - someone may have answered your question already.

### Reporting Bugs and Making Suggestions
Upon identifying any bugs or features that could be improved, please open an
[issue](https://github.com/CITCOM-project/CausalTestingFramework/issues) and label with bug or feature suggestion. Every issue
should clearly explain the bug or feature to be improved and, where necessary, instructions to replicate. We also
provide templates for each scenario when creating an issue.

### Contributing to the Codebase
To contribute to our work, please ensure the following:

1. [Fork the repository](https://help.github.com/articles/fork-a-repo/) into your own GitHub account, and [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) it to your local machine.
2. [Create a new branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) in your forked repository. Give this branch an appropriate name, and create commits that describe the changes.
3. [Push your changes](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository) to your new branch in your remote fork, compare with `CausalTestingFramework/main`, and ensure any conflicts are resolved.
4. Create a draft [pull request](https://docs.github.com/en/get-started/quickstart/hello-world#opening-a-pull-request) from your branch, and ensure you have [linked](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls) it to any relevant issues in your description.

### Continuous Integration (CI) and Code Quality
Our CI/CD tests include:
    - Build: Install the necessary Python and runtime dependencies.
    - Linting: [pylint](https://pypi.org/project/pylint/) is employed for our code linter and analyser.
    - Testing: We use the [unittest]() module to develop our tests and the [pytest](https://pytest.org/en/latest/) framework as our test discovery.
    - Formatting: We use [black](https://pypi.org/project/black/) for our code formatting.

To find the other (optional) developer dependencies, please check `pyproject.toml`.

### Pre-commit Hooks
We use [pre-commit](https://pre-commit.com/) to automatically run code quality checks before each commit. This ensures consistent code style and catches issues early.

Automated checks include:

- Trailing whitespace removal.
- End-of-file fixing.
- YAML and TOML validation.
- Black formatting.
- isort import sorting.
- Pylint code analysis.

To use pre-commit:
```bash
# Install pre-commit hooks (one-time setup of .pre-commit-config.yaml)
pre-commit install

# Manually run hooks on all files (optional)
pre-commit run --all-files
```

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
9. Code formatting and linting is handled automatically by pre-commit hooks (see above).

### Manual Code Quality Checks (Optional)
While pre-commit handles most formatting automatically, you can run these commands manually if needed:

```bash
# Format code
black causal_testing

# Sort imports
isort causal_testing

# Run linter
pylint causal_testing

# Run tests
pytest
```

### Compatibility Testing Across Python Versions with tox

For compatibility testing Python versions, we use [tox](https://pypi.org/project/tox/) to automate
testing across all supported Python versions (3.10, 3.11, 3.12, and 3.13):

- Install tox: `pip install tox`.
- Test all versions: `tox` (runs tests on all Python versions + linting in the root folder).
- Test specific version: `tox -e py313` (or py310, py311, py312).
- Quick iteration: Use `pytest` directly for fast testing during development.
