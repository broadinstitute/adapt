"""Setup script for the dxguidedesign distribution using Distutils.
"""

from setuptools import find_packages
from setuptools import setup

import dxguidedesign

__author__ = 'Hayden Metsky <hayden@mit.edu>'


setup(name='dxguidedesign',
      description='Tools to design guides for diagnostics',
      author='Hayden Metsky',
      author_email='hayden@mit.edu',
      packages=find_packages(),
      install_requires=['numpy>=1.9.0', 'scipy>=1.0.0'],
      scripts=[
        'bin/design.py',
        'bin/analyze_coverage.py'
      ])
