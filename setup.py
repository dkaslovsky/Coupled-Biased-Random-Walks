import os
import re
from setuptools import setup


dir_name = os.path.dirname(__file__)

readme_path = os.path.join(dir_name, 'README.md')
with open(readme_path, 'r') as f:
    long_description = f.read()

requirements_path = os.path.join(dir_name, 'requirements.txt')
with open(requirements_path, 'r') as f:
    requirements = f.read().splitlines()

version_path = os.path.join('coupled_biased_random_walks', '_version.py')
with open(version_path, 'r') as f:
    version = f.read()


def parse_version(version_str: str) -> str:
    """
    Parse the package version
    :param version_str: string of the form "__version__ = 'x.y.z'"
    """
    version_str = version_str.strip('\n')
    try:
        version = version_str.split('=')[1]
    except IndexError:
        raise RuntimeError(f'cannot parse version from [{version_str}]')
    version = version.strip(' \'\"')
    # validate version
    if not re.match(r'^(\d+).(\d+).(\d+)$', version):
        raise RuntimeError(f'parsed invalid version [{version}] from [{version_str}]')
    return version


setup(
    name='coupled_biased_random_walks',
    version=parse_version(version),
    author='Daniel Kaslovsky',
    author_email='dkaslovsky@gmail.com',
    license='MIT',
    description='Outlier detection for categorical data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['coupled_biased_random_walks'],
    install_requires=requirements,
    url='https://github.com/dkaslovsky/Coupled-Biased-Random-Walks',
    keywords=['anomaly detection, outlier detection', 'categorical data', 'random walk'],
    classifiers=[
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
