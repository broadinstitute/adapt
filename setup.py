"""Setup script for the probedesign distribution using Distutils.
"""

from setuptools import find_packages
from setuptools import setup

import probedesign

__author__ = 'Hayden Metsky <hayden@mit.edu>'


setup(name='probedesign',
      description='Tools to design probes for diagnostics',
      author='Hayden Metsky',
      author_email='hayden@mit.edu',
      packages=find_packages(),
      scripts=[
        'bin/design_probes.py'
      ])
