"""Setup script for the adapt distribution using Distutils.
"""

from setuptools import find_packages
from setuptools import setup

import adapt

__author__ = 'Hayden Metsky <hayden@mit.edu>'


setup(name='adapt',
      description='Tools to design guides for diagnostics',
      author='Hayden Metsky',
      author_email='hayden@mit.edu',
      packages=find_packages(),
      install_requires=['numpy>=1.16.0', 'scipy>=1.4.0', 'tensorflow>=2.1.0'],
      scripts=[
        'bin/design.py',
        'bin/design_naively.py',
        'bin/analyze_coverage.py',
        'bin/pick_test_targets.py'
      ])
