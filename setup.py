from setuptools import setup, find_packages

requirements = [
    "numpy~=1.20",
    "pandas~=1.3.4",
    "setuptools~=58.5.3",
    "networkx~=2.6.3",
    "pygraphviz~=1.7",
    "pytest~=6.2.5",
    "scikit-learn~=1.0.1",
    "matplotlib~=3.5.0",
    "econml~=0.12.0",
    "statsmodels~=0.13.1",
]

setup(
    name="causal_testing_framework",
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages(),
)
