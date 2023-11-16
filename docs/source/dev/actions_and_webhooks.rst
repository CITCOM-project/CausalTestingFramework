Github Actions and Webhooks
===========================

Actions
--------------

Currently, this project makes use of 3 `Github Actions <https://github.com/features/actions>`_,
which can be found in the
`.github/workflows <https://github.com/CITCOM-project/CausalTestingFramework/tree/main/.github/workflows>`_ directory.

They are:

#.  ``ci-tests.yaml``, which runs continuous integration (CI) tests on each on each pull request.

#.  ``lint-format.yaml``, which runs linting on each pull request.

#.  ``publish-to-pypi.yaml``, runs when a new version tag is pushed and publishes that tag version to PyPI.

Webhooks
---------------

The project also uses 3 `Webhooks <https://docs.github.com/en/webhooks-and-events/webhooks/about-webhooks>`_, which can
be found in the `project settings <https://github.com/CITCOM-project/CausalTestingFramework/settings>`_ on Github. These
include:

#.  `Codacy <https://github.com/codacy>`_

#.  `Codecov <https://github.com/codecov>`_

#.  `Read the Docs <https://github.com/readthedocs>`_