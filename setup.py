from setuptools import setup

requirements = [
    'numpy~=1.21.4',
    'pandas~=1.3.4',
    'setuptools~=58.5.3',
    'networkx~=2.6.3',
    'pygraphviz~=1.7',
    'dowhy~=0.6',
    'rpy2~=3.4.5'
]

setup(
    name='causaltest',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'myscript=myscript:run'
        ]
    }
)
