from setuptools import setup, find_packages

setup(
    name="causaltest",
    version="0.0.1",
    url="https://github.com/CITCOM-project/CausalTestingFramework",
    author="CITCoM Team",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "networkx", "pygraphviz", "dowhy", "rpy2",],
)
