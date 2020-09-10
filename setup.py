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
      install_requires=['numpy>=1.16.0,<1.19.0', 'scipy==1.4.1', 'tensorflow>=2.3.0'],
      extras_require={
        'AWS': ['boto3>=1.14.54', 'botocore>=1.17.54']
      },
      scripts=[
        'bin/design.py',
        'bin/design_naively.py',
        'bin/analyze_coverage.py',
        'bin/pick_test_targets.py'
      ])
