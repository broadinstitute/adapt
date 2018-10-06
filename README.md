diagnostic-guide-design
=======================

This is a Python package for designing guides to be used for diagnostics with SHERLOCK.

## Dependencies

It is tested under Python 3.5, but should also work with earlier versions of Python 3. There are no other dependencies.

## Install

You can install the package with `pip`:
```
git clone git@github.com:broadinstitute/diagnostic-guide-design.git
cd diagnostic-guide-design
pip install -e .
```

## Testing

The package uses Python's `unittest` framework. To execute all unit tests, run:
```
python -m unittest discover
```

## Running

To see the arguments that the program accepts, run:
```
python bin/design_guides.py -h
```
Note that output sequences are directly from the input sequences; guide sequences should be reverse complements of the output.

## Examples

The package includes an alignment of LASV sequences (S segment) from Sierra Leone. For example:
```
python bin/design_guides.py examples/SLE_S.aligned.fasta -o guides.tsv -w 200 -l 28 -m 1 -p 0.95
```
reads an alignment from `examples/SLE_S.aligned.fasta`. From this alignment, it scans each 200 nt window (`-w 200`) to find the smallest collection of guides that:
* are all within the window
* are 28 nt long (`-l 28`)
* capture 95% of all input sequences (`-p 0.95`) tolerating up to 1 mismatch (`-m 1`)


It outputs a TSV file, `guides.tsv`, in which each row corresponds to a window in the alignment and the columns give information about the guides designed for that window. The columns are:
* `window-start`/`window-end`: start (inclusive) and end (exclusive) positions of this window in the alignment
* `count`: the number of guide sequences for this window
* `score`: a statistic between 0 and 1 that describes the redundancy of the guide sequences in capturing the input sequences (higher is better); it is meant to break ties between windows that have the same number of guide sequences, and is not intended to be compared between windows with different numbers of guide sequences
* `target-sequences`: the sequences of the targets for this window from which to construct guides, separated by spaces (guides should be reverse complements of these sequences)
* `target-sequence-positions`: the positions of the guide sequences in the alignment, in the same order as the sequences are reported; since a guide may come from >1 position, positions are reported in set notation (e.g., \{100\})

By default, the rows in `guides.tsv` are sorted by the position of the window. If you include the `--sort` argument to the program, it will sort the rows in `guides.tsv` so that the "best" choices of windows are on top. It sorts by `count` (ascending) followed by `score` (descending), so that windows with the fewest guides and highest score are on top.
