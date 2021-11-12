from setuptools import setup
setup(
    name='causaltest',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'myscript=myscript:run'
        ]
    }
)