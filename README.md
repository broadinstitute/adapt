# ADAPT &nbsp;&middot;&nbsp; [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/broadinstitute/adapt/pulls) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
#### Adaptive Design by Astutely Patrolling Targets

ADAPT is a software package for designing optimal nucleic acid diagnostics.

* **End-to-end**: ADAPT connects directly with publicly available genome databases, downloading and curating their data, so designs can be made rapidly and always account for the latest known microbial diversity.
* **Comprehensive**: ADAPT optimally accounts for the full scope of known sequence diversity across input taxa, so designs are both minimal in size and effective against variable targets.
* **Sensitive**: ADAPT can incorporate a predictive model of activity, so designs are predicted to be highly active against targets.
* **Specific**: ADAPT enforces high specificity against set non-targeted taxa, so designs are more likely to be accurate in distinguishing between related taxa.

The methods and software are not yet published.
However, the problems share some similarity with the problems solved by CATCH, which is described in [_Nature Biotechnology_](https://www.nature.com/articles/s41587-018-0006-x) and available publicly on [GitHub](https://github.com/broadinstitute/catch).
<br/>
<br/>

### Table of contents

* [Setting up ADAPT](#setting-up-adapt)
  * [Dependencies](#dependencies)
  * [Setting up a conda environment](#setting-up-a-conda-environment)
  * [Downloading and installing](#downloading-and-installing)
  * [Testing](#testing)
* [Using ADAPT](#using-adapt)
  * [Designing guides](#designing-guides)
  * [Common options](#common-options)
  * [Output](#output)
* [Examples](#examples)
  * [Designing with sliding window against a single target](#designing-with-sliding-window-against-a-single-target)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
<br/>

# Setting up ADAPT

## Dependencies

ADAPT requires:
* [Python](https://www.python.org) &gt;= 3.5
* [NumPy](http://www.numpy.org) &gt;= 1.9.0
* [SciPy](https://www.scipy.org) &gt;= 1.0.0

Installing ADAPT with `pip`, as described below, will install NumPy and SciPy if they are not already installed.

If using alignment features in subcommands below, ADAPT also requires a path to an executable of [MAFFT](https://mafft.cbrc.jp/alignment/software/).

## Setting up a conda environment

_Note: This section is optional, but may be useful to users who are new to Python._

It is generally useful to install and run Python packages inside of a [virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment), especially if you have multiple versions of Python installed or use multiple packages.
This can prevent problems when upgrading, conflicts between packages with different requirements, installation issues that arise from having different Python versions available, and more.

One option to manage packages and environments is to use [conda](https://conda.io/en/latest/).
A fast way to obtain conda is to install Miniconda: you can download it [here](https://conda.io/en/latest/miniconda.html) and find installation instructions for it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
For example, on Linux you would run:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Once you have conda, you can [create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) an environment for ADAPT with Python 3.7:
```bash
conda create -n adapt python=3.7
```
Then, you can activate the `adapt` environment:
```bash
conda activate adapt
```
After the environment is created and activated, you can install ADAPT as described below.
You will need to activate the environment each time you use ADAPT.

## Downloading and installing

An easy way to setup ADAPT is to clone the repository and install the package with `pip`:
```bash
git clone git@github.com:broadinstitute/adapt.git
cd adapt
pip install -e .
```
Depending on your setup (i.e., if you do not have write permissions in the installation directory), you may need to supply `--user` to `pip install`.

## Testing

The package uses Python's `unittest` framework.
To execute all unit tests, run:
```bash
python -m unittest discover
```

# Using ADAPT

## Designing guides

The main program to design guides is [`design.py`](./bin/design.py).

[`design.py`](./bin/design.py) requires two subcommands:
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] ...
```

SEARCH-TYPE is one of:

* `sliding-window`: Search for crRNAs within a sliding window of a fixed size, and output an optimal crRNA set for each window.
* `complete-targets`: Search for primer pairs and crRNAs between them.
Output the top _N_ targets, where each target contains primer pairs and crRNAs between them.

INPUT-TYPE is one of:

* `fasta`: The input is one or more FASTA files, each containing aligned sequences for a taxon.
If more than one file is provided, the search performs differential identification across the taxa.
* `auto-from-file`: The input is a file containing a list of taxon IDs and related information.
This fetches sequences for those taxa, then curates, clusters and aligns the sequences for each taxon, and finally uses the generated alignments as input for design.
The search finds crRNAs for differential identification across the taxa.
* `auto-from-args`: The input is a single taxonomic ID, and related information, provided as command-line arguments.
This fetches sequences for the taxon, then curates, clusters and aligns the sequences, and finally uses the generated alignment as input for design.

To see details on all the arguments to use for a particular choice of subcommands, run:
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] --help
```
This includes required positional arguments for each choice of subcommands.

## Common options

Below is a summary of some common arguments to [`design.py`](./bin/design.py):

* `-gl GUIDE_LENGTH`: Design guides to be GUIDE_LENGTH nt long.
(Default: 28.)
* `-gm MISMATCHES`: Tolerate up to MISMATCHES mismatches when determining whether a guide hybridizes to a sequence.
(Default: 0.)
* `-gp COVER_FRAC`: Design guides such that at least a fraction COVER_FRAC of the genomes are hit by the guides.
(Default: 1.0.)
* `--cover-by-year-decay YEAR_TSV MIN_YEAR_WITH_COVG DECAY`: Group input sequences by year and set a separate desired COVER_FRAC for each year.
See `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on this argument.
Note that when INPUT-TYPE is `auto-from-{file,args}`, this argument does not accept YEAR_TSV.
* `--id-m ID_M` / `--id-frac ID_FRAC`: Design guides to perform differential identification where these parameters determine specificity.
Allow for up to ID_M mismatches when determining whether a guide hits a sequence in a taxon other than the one for which it is being designed, and decide that a guide hits a taxon if it hits at least ID_FRAC of the sequences in that taxon.
ADAPT does not output guides that hit group/taxons other than the one for which they are being designed.
Higher values of ID_M and lower values of ID_FRAC correspond to more specificity.
(Default: 2 for ID_M, 0.05 for ID_FRAC.)
* `--specific-against [alignment] [alignment ...]`: Design guides to be specific against the provided alignments (in FASTA format).
That is, the guides should not hit sequences in these FASTA files, as measured by ID_M and ID_FRAC.
* `--do-not-allow-gu-pairing`: If set, do not count G-U (wobble) base pairs between guide and target sequence as matching.
* `--require-flanking5 REQUIRE_FLANKING5` / `--require-flanking3 REQUIRE_FLANKING3`: Require the given sequence on the 5' (REQUIRE_FLANKING5) and/or 3' (REQUIRE_FLANKING3) protospacer flanking site (PFS) for each designed guide.
This tolerates ambiguity in the sequence (e.g., 'H' requires 'A', 'C', or 'T').
* `--required-guides REQUIRED_GUIDES`: Ensure that the guides provided in REQUIRED_GUIDES are included in the design, and perform the design with them already included.
See `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on the REQUIRED_GUIDES file format.
* `--blacklisted-ranges BLACKLISTED_RANGES`: Do not construct guides in the ranges provided in BLACKLISTED_RANGES.
* `--blacklisted-kmers BLACKLISTED_KMERS`: Do not construct guides that contain k-mers provided in BLACKLISTED_KMERS.

Below are some additional arguments when SEARCH-TYPE is `complete-targets`:

* `-pl PRIMER_LENGTH`: Design primers to be PRIMER_LENGTH nt long.
(Default: 30.)
* `-pp PRIMER_COVER_FRAC`: Same as `-gp`, except for the design of primers.
(Default: 1.0.)
* `-pm PRIMER_MISMATCHES`: Tolerate up to PRIMER_MISMATCHES mismatches when determining whether a primer hybridizes to a sequence.
(Default: 0.)
* `--max-primers-at-site MAX_PRIMERS_AT_SITE`: Only allow up to MAX_PRIMERS_ATE_SITE primers at each primer set.
If not set, there is no limit.
Smaller values can significantly improve runtime.
(Default: not set.)
* `--cost-fn-weights COST_FN_WEIGHTS`: Coefficients to use in a cost function for each target.
See `design.py complete-targets [INPUT-TYPE] --help` for details.
* `--best-n-targets BEST_N_TARGETS`: Only compute and output the best BEST_N_TARGETS targets, where each target receives a cost according to COST_FN_WEIGHTS.
Note that higher values can significantly increase runtime.
(Default: 10.)

Below are some additional arguments when INPUT-TYPE is `auto-from-{file,args}`:

* `--mafft-path MAFFT_PATH`: Use the [MAFFT](https://mafft.cbrc.jp/alignment/software/) executable at MAFFT_PATH for generating alignments.
* `--prep-memoize-dir PREP_MEMOIZE_DIR`: Memoize alignments and statistics on these alignments in PREP_MEMOIZE_DIR.
If not set (default), do not memoize this information.
If repeatedly re-running on the same taxonomies, using this can significantly improve runtime.
* `--prep-influenza`: If set, use NCBI's influenza database for fetching data.
This must be specified if design is for influenza A/B/C viruses.
* `--sample-seqs SAMPLE_SEQS`: Randomly sample SAMPLE_SEQS accessions with replacement from each taxonomy, and move forward with the design using this sample.
This can be useful for measuring various properties of the design.
* `--cluster-threshold CLUSTER_THRESHOLD`: Use CLUSTER_THRESHOLD as the maximum inter-cluster distance when clustering sequences prior to alignment.
The distance is average nucleotide dissimilarity (1-ANI); higher values result in fewer clusters.
(Default: 0.2.)
* `--use-accessions USE_ACCESSIONS`: Use the specified accessions, in a file at the path USE_ACCESSIONS, for generating input.
This is performed instead of fetching neighbors from NCBI.
See `design.py [SEARCH-TYPE] auto-from-{file,args} --help` for details on the format of the file.

## Output

The files output by ADAPT are TSV files, but vary in format depending on SEARCH-TYPE and INPUT-TYPE.
There is a separate TSV file for each taxon.

For all cases, see `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on the output format and how to specify paths to the output TSV files.

### Sliding window

When SEARCH-TYPE is `sliding-window`, each row corresponds to a window in the alignment and the columns give information about the guides designed for that window.
The columns are:
* `window-start`/`window-end`: Start (inclusive) and end (exclusive) positions of this window in the alignment.
* `count`: The number of guide sequences for this window.
* `score`: A statistic between 0 and 1 that describes the redundancy of the guide sequences in capturing the input sequences (higher is better); it is meant to break ties between windows that have the same number of guide sequences, and is not intended to be compared between windows with different numbers of guide sequences.
* `total-frac-bound`: The total fraction of all sequences in the alignment that are bound by a guide. Note that if `--cover-by-year-decay` is provided, this might be considerably less than COVER_FRAC.
* `target-sequences`: The sequences of the targets for this window from which to construct guides, separated by spaces (guides should be reverse complements of these sequences).
* `target-sequence-positions`: The positions of the guide sequences in the alignment, in the same order as the sequences are reported; since a guide may come from >1 position, positions are reported in set notation (e.g., \{100\}).

By default, when SEARCH-TYPE is `sliding-window`, the rows in the output are sorted by the position of the window.
If you include the `--sort` argument to [`design.py`](./bin/design.py), it will sort the rows in the output so that the "best" choices of windows are on top.
It sorts by `count` (ascending) followed by `score` (descending), so that windows with the fewest guides and highest score are on top.

### Complete targets

When SEARCH-TYPE is `complete-targets`, each row is a possible target (primer pair and crRNA combination) and there are additional columns giving information about primer pairs.
There is also a `cost` column, giving the cost of each target according to `--cost-fn-weights`.
The rows in the output are sorted by the cost (ascending, so that better targets are on top).

When INPUT-TYPE is `auto-from-file` or `auto-from-args`, there is a separate TSV file for each cluster of input sequences.

### Complementarity

Note that output sequences are in the same sense as the input sequences.
Synthesized guide sequences should be reverse complements of the output!

# Examples

## Designing with sliding window against a single target

This is the most simple example.
The package includes an alignment of LASV sequences (S segment) from Sierra Leone.
For example:
```bash
design.py sliding-window fasta examples/SLE_S.aligned.fasta -o guides.tsv -w 200 -gl 28 -gm 1 -gp 0.95
```
reads an alignment from `examples/SLE_S.aligned.fasta`.

From this alignment, it scans each 200 nt window (`-w 200`) to find the smallest collection of guides that:
* are all within the window
* are 28 nt long (`-gl 28`)
* capture 95% of all input sequences (`-gp 0.95`) tolerating up to 1 mismatch (`-gm 1`)

It outputs a file, `guides.tsv`, that contains constructed guide sequences.
See [Output](#output) above for a description of this file.

# Contributing

We welcome contributions to ADAPT.
This can be in the form of an [issue](https://github.com/broadinstitute/adapt/issues) or [pull request](https://github.com/broadinstitute/adapt/pulls).
If you have questions, please create an [issue](https://github.com/broadinstitute/adapt/issues) or email **Hayden Metsky** &lt;hayden@mit.edu&gt;.

# Citation

ADAPT is not yet published.
If you find it useful to your work, please let us know and inquire about how to cite it.

# License

ADAPT is licensed under the terms of the [MIT license](./LICENSE).
