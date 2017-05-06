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
      scripts=[
        'bin/design_guides.py'
      ])
