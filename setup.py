from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='wqsat_format',
    packages=find_packages(),
    version='0.1.0',
    description='A Python package for processing Sentinel-2 and Sentinel-3 images.',
    author='CSIC',
    license='Apache License 2.0',
    install_requires=reqs)
