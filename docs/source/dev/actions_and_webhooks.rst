Github Actions and Webhooks
===========================

Github Actions
--------------

Currently this project makes use of 3 `Github Actions <https://github.com/features/actions>`_, which can be found in the `.github/workflows <https://github.com/CITCOM-project/CausalTestingFramework/tree/main/.github/workflows>`_ directory

They are:

#.  ci-tests.yaml, which runs CI tests on each PR

#.  lint-format.yaml, runs linting on each PR

#.  publish-to-pypi.yaml, runs when a new version tag is pushed and publishes that tag version to pypi

Github Webhooks
---------------

The project also uses 3 `Webhooks <https://docs.github.com/en/webhooks-and-events/webhooks/about-webhooks>`_, which can be found in the `project settings <https://github.com/CITCOM-project/CausalTestingFramework/settings>`_ on Github

#.  To codacy

#.  To codecov

#.  To readthedocs