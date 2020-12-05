"""Setup script for the adapt distribution using Distutils.
"""

from setuptools import find_packages
from setuptools import setup

import adapt

__author__ = 'Hayden Metsky <hayden@mit.edu>'

LICENSE = """MIT License

Copyright (c) 2020 Broad Institute, Inc. and Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

setup(name='adapt',
      description='Tools to design guides for diagnostics',
      author='Hayden Metsky',
      author_email='hayden@mit.edu',
      version=adapt.__version__,
      license=LICENSE,
      packages=find_packages(),
      install_requires=['numpy>=1.16.0,<1.19.0', 'scipy==1.4.1', 'tensorflow==2.3.0'],
      extras_require={
        'AWS': ['boto3>=1.14.54', 'botocore>=1.17.54']
      },
      scripts=[
        'bin/design.py',
        'bin/design_naively.py',
        'bin/analyze_coverage.py',
        'bin/pick_test_targets.py'
      ])
