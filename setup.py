from setuptools import setup, find_packages

requirements = [
    "econml~=0.12.0",
    "fitter~=1.4.0",
    "lhsmdu~=1.1",
    "networkx~=2.6.3",
    "numpy~=1.22.4",
    "pandas~=1.3.5",
    "scikit_learn~=1.1.2",
    "scipy~=1.7.3",
    "statsmodels~=0.13.2",
    "tabulate~=0.8.10",
    "z3_solver~=4.11.2",
]

# Additional dependencies for development
dev_requirements = [
    "autopep8",
    "isort",
    "pytest",
    "pylint",
    "black",
    "autoapi",
    "myst-parser",
    "sphinx-autoapi",
    "sphinx_rtd_theme",
]

readme = open("README.md", encoding="UTF-8").read()

setup(
    name="causal_testing_framework",
    version="1.0.0",
    description="A framework for causal testing using causal directed acyclic graphs.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The CITCOM team",
    url="https://github.com/CITCOM-project/CausalTestingFramework",
    project_urls={
        "Bug Tracker": "https://github.com/CITCOM-project/CausalTestingFramework/issues",
        "Documentation": "https://causal-testing-framework.readthedocs.io/",
        "Source": "https://github.com/CITCOM-project/CausalTestingFramework",
    },
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=find_packages(),
    license="MIT",
    keywords="causal inference, verification",
)
