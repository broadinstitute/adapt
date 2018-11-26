diagnostic-guide-design
=======================

*dgd* is a Python package for designing guides to be used for diagnostics with SHERLOCK.
<br/>

### Table of contents

* [Setting up dgd](#setting-up-dgd)
  * [Dependencies](#dependencies)
  * [Downloading and installing](#downloading-and-installing)
  * [Testing](#testing)
* [Using dgd](#using-dgd)
  * [Designing guides](#designing-guides)
  * [Additional options](#additional-options)
  * [Output](#output)
* [Examples](#examples)
  * [Designing against a single target](#designing-against-a-single-target)
<br/>

# Setting up dgd

## Dependencies

dgd is tested using Python 3.5, but should also work with earlier versions of Python 3.
There are no other dependencies.

## Downloading and installing

An easy way to setup dgd is to clone the repository and install the package with `pip`:
```bash
git clone git@github.com:broadinstitute/diagnostic-guide-design.git
cd diagnostic-guide-design
pip install --user -e .
```

## Testing

The package uses Python's `unittest` framework.
To execute all unit tests, run:
```bash
python -m unittest discover
```

# Using dgd

## Designing guides

The main program to design guides is [`design_guides.py`](./bin/design_guides.py).
To see details on all the arguments that the program accepts, run:
```bash
design_guides.py --help
```

[`design_guides.py`](./bin/design_guides.py) requires paths one or more input files, and corresponding output files:

```bash
design_guides.py [input] [input ...] -o [output] [output ...]
```

No other arguments are required.
Each `input` is a path to a FASTA file that consists of an alignment of sequences from which to design guides.
Each `output` is a path to a TSV file that will be written, containing the designed guide sequences.
The _i_'th output corresponds to the _i_'th input, so there must be the same number of outputs as inputs.
Unless performing differential identification with `--id` (see below), this treats each input completely independently of the others.

## Additional options

Below is a summary of some useful arguments to [`design_guides.py`](./bin/design_guides.py):

* `-l GUIDE_LENGTH`: Design guides to be GUIDE_LENGTH nt long.
(Default: 28.)
* `-m MISMATCHES`: Tolerate up to MISMATCHES mismatches when determining whether a guide covers a sequence.
(Default: 0.)
* `-w WINDOW_SIZE`: Ensure that all designed guides are within a window of this size.
Guides are designed separately for each window of length WINDOW_SIZE nt.
(Default: 200.)
* `-p COVER_FRAC`: Design guides such that at least a fraction COVER_FRAC of the genomes are hit by the guides.
(Default: 1.0.)
* `--cover-by-year-decay YEAR_TSV MIN_YEAR_WITH_COVG DECAY`: Group input sequences by year and set a separate desired COVER_FRAC for each year.
See `design_guides.py --help` for details on this argument.
* `--id` / `--id-m ID_M` / `--id-frac ID_FRAC`: Design guides to perform differential identification, in which each input FASTA is a group/taxon to identify with specificity.
Allow for up to ID_M mismatches when determining whether a guide hits a sequence in a group/taxon other than the one for which it is being designed, and decide that a guide hits a group/taxon if it hits at least ID_FRAC of the sequences in that group/taxon.
dgd does not output guides that hit group/taxons other than the one for which they are being designed.
Higher values of ID_M and lower values of ID_FRAC correspond to more specificity.
Note that `--id` must be set to perform differential identification (setting `--id-m` and/or `--id-frac` alone will not suffice).
(Default: 2 for ID_M, 0.05 for ID_FRAC.)
* `--do-not-allow-gu-pairing`: If set, do not count G-U (wobble) base pairs as matching.
* `--required-guides REQUIRED_GUIDES`: Ensure that the guides provided in REQUIRED_GUIDES are included in the design, and perform the design with them already included.
See `design_guides.py --help` for details on the REQUIRED_GUIDES file format.
* `--blacklisted-ranges BLACKLISTED_RANGES`: Do not construct guides in the ranges provided in BLACKLISTED_RANGES.
See `design_guides.py --help` for details on the BLACKLISTED_RANGES file format.
* `--blacklisted-kmers BLACKLISTED_KMERS`: Do not construct guides that contain k-mers provided in BLACKLISTED_KMERS.
See `design_guides.py --help` for details on the BLACKLISTED_KMERS file format.

## Output

Each file output by dgd is a TSV file in which each row corresponds to a window in the alignment and the columns give information about the guides designed for that window.
The columns are:
* `window-start`/`window-end`: Start (inclusive) and end (exclusive) positions of this window in the alignment.
* `count`: The number of guide sequences for this window.
* `score`: A statistic between 0 and 1 that describes the redundancy of the guide sequences in capturing the input sequences (higher is better); it is meant to break ties between windows that have the same number of guide sequences, and is not intended to be compared between windows with different numbers of guide sequences.
* `total-frac-bound`: The total fraction of all sequences in the alignment that are bound by a guide. Note that if `--cover-by-year-decay` is provided, this might be considerably less than COVER_FRAC.
* `target-sequences`: The sequences of the targets for this window from which to construct guides, separated by spaces (guides should be reverse complements of these sequences).
* `target-sequence-positions`: The positions of the guide sequences in the alignment, in the same order as the sequences are reported; since a guide may come from >1 position, positions are reported in set notation (e.g., \{100\}).

By default, the rows in the output are sorted by the position of the window.
If you include the `--sort` argument to [`design_guides.py`](./bin/design_guides.py), it will sort the rows in the output so that the "best" choices of windows are on top.
It sorts by `count` (ascending) followed by `score` (descending), so that windows with the fewest guides and highest score are on top.

Note that output sequences are directly from the input sequences; guide sequences should be reverse complements of the output!

# Examples

## Designing against a single target

The package includes an alignment of LASV sequences (S segment) from Sierra Leone.
For example:
```bash
design_guides.py examples/SLE_S.aligned.fasta -o guides.tsv -w 200 -l 28 -m 1 -p 0.95
```
reads an alignment from `examples/SLE_S.aligned.fasta`.

From this alignment, it scans each 200 nt window (`-w 200`) to find the smallest collection of guides that:
* are all within the window
* are 28 nt long (`-l 28`)
* capture 95% of all input sequences (`-p 0.95`) tolerating up to 1 mismatch (`-m 1`)

It outputs a file, `guides.tsv`, that contain constructed guide sequences.
See [Output](#output) above for a description of this file.
