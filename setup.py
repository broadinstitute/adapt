"""Setup script for the adapt distribution using Distutils.
"""

from setuptools import find_packages
from setuptools import setup

import adapt

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

with open("README.md", "r", encoding="utf-8") as f:
  LONG_DESCRIPTION = f.read()

# There is a PyPi package called 'adapt' already, so renamed for PyPi
# Does not affect the Bioconda package name
setup(name='adapt-diagnostics',
      description='Tools to design guides for diagnostics',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author='Hayden Metsky',
      author_email='hmetsky@broadinstitute.org',
      maintainer='Priya P. Pillai',
      maintainer_email='ppillai@broadinstitute.org',
      url="https://github.com/broadinstitute/adapt",
      version=adapt.__version__,
      packages=find_packages(),
      install_requires=['numpy>=1.16.0,<1.19.0', 'scipy==1.4.1', 'tensorflow==2.3.0'],
      extras_require={
        'AWS': ['boto3>=1.14.54', 'botocore>=1.17.54']
      },
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      scripts=[
        'bin/design.py',
        'bin/design_naively.py',
        'bin/analyze_coverage.py',
        'bin/pick_test_targets.py'
      ])
