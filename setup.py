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
    "z3_solver~=4.8.13.0",
]

# Additional dependencies for development
dev_requirements = ["autopep8", "isort", "pytest", "pylint", "black"]

setup(
    name="causal_testing_framework",
    version="0.0.1",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=find_packages(),
)
