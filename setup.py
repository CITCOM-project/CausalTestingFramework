from setuptools import setup, find_packages

requirements = [
    'numpy~=1.21.4',
    'pandas~=1.3.4',
    'setuptools~=58.5.3',
    'networkx~=2.6.3',
    'pygraphviz~=1.7',
    'pytest~=6.2.5'
]

setup(
    name='causal_testing_framework',
    version='0.0.1',
    packages=find_packages()
)
