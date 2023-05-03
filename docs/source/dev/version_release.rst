Version Releases
================

This project follows the `Semantic Versioning 2.0.0 <https://semver.org/>`_ style for code releases.
This page aims to describe the release process for a new code version on the `Project Github <https://github.com/CITCOM-project/CausalTestingFramework>`_

When to release
---------------

A new release should follow each successful PR merge, or group of related PR merges.

How to release
--------------

#. Once your PR(s) are merged, navigate to https://github.com/CITCOM-project/CausalTestingFramework/releases, which can be
found on the right hand side of the projects Github main page by clicking on 'Releases'

#. Press the **Draft a new release** button in the top right of the releases page

#. Press the **Choose a tag** button and add the new version following the Semantic Version guidelines. Please include the 'v' before the tag, e.g. **v0.0.0**

#. Enter the same tag name into the **Release Title** box

#. Press **Generate Release Notes** button

#. Add any additional information that may be helpful to the release notes. If there are breaking changes for example, which modules will they affect?

#. Ensure the **Set as the latest release** checkbox is selected

#. Press publish release

From here, a `Github action <https://github.com/CITCOM-project/CausalTestingFramework/blob/main/.github/workflows/publish-to-pypi.yaml>`_ will then push this release to PyPI where it can be installed using the usual pip version commands. e.g. ``pip install causal-testing-framework==0.0.0``