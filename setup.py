from distutils.core import setup

from os import path
from io import open


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()


setup(
    name='coupled_biased_random_walks',
    version='0.1',
    packages=['coupled_biased_random_walks'],
    license='MIT',
    author='Daniel Kaslovsky',
    author_email='dkaslovsky@gmail.com',
    description='Outlier detection for categorical data',
    long_description=long_description,
    keywords=['anomaly detection, outlier detection', 'categorical data', 'random walk'],
    install_requires=requirements
)
