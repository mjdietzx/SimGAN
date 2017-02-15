import os

from setuptools import find_packages
from setuptools import setup


def read(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), 'r') as f:
        return f.readlines()

setup(
    name='SimGAN',
    version='0.0.0',
    description='Implementation of Apple\'s Learning from Simulated and Unsupervised Images through Adversarial '
                'Training.',
    # long_description=read('README.md'),
    url='https://github.com/wayaai/SimGAN.git',
    author='Michael Dietz',
    keywords='',
    packages=find_packages(exclude=["tests.*", "tests"]))
