#!/usr/bin/env python
"""NeuralPixels
Is an open source library to aide in neural network development and training
"""

import setuptools

import os
import sys


if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

with open(readme_path, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neuralpixels',
    version='0.1',
    scripts=['neuralpixels'],
    author="Jaret Burkett",
    author_email="jaret@neuralpixels.com",
    description="Official NeuralPixels package to aide in machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuralpixels/neuralpixels",
    packages=setuptools.find_packages(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
