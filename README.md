diagnostic-guide-design
=======================

This is a Python package for designing guides to be used for diagnostics with SHERLOCK.

## Dependencies

It is tested under Python 3.5, but should also work with earlier versions of Python 3. There are no other dependencies.

## Install

You can install the package with `pip`:
```
$ git clone https://github.com/broadinstitute/diagnostic-guide-design.git
$ cd diagnostic-guide-design
$ pip install -e .
```

## Testing

The package uses Python's `unittest` framework. To execute all unit tests, run:
```
$ python -m unittest discover
```

## Running

To see the arguments that the program accepts, run:
```
$ python bin/design_guides.py -h
```

