diagnostic-guide-design
=======================

This is a Python package for designing guides to be used for diagnostics with SHERLOCK.

## Dependencies

It is tested under Python 3.5, but should also work with earlier versions of Python 3. There are no other dependencies.

## Install

You can install the package with `pip`:
```
$ git clone git@github.com:broadinstitute/diagnostic-guide-design.git
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

## Examples

The package includes an alignment of LASV sequences (S segment) from Sierra Leone. For example:
```
$ python bin/design_guides.py examples/SLE_S.aligned.fasta -l 28 -w 200 -m 1 -p 0.95
```
will output 2 guides that:
* are 28 nt long (`-l 28`)
* are within a 200 nt window (`-w 200`)
* capture 95% of all input sequences (`-p 0.95`) tolerating up to 1 mismatch (`-m 1`)

