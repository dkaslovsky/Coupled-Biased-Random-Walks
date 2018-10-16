from setuptools import setup


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='coupled_biased_random_walks',
    version='1.0.0',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
