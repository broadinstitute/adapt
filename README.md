# ADAPT &nbsp;&middot;&nbsp; [![Build Status](https://travis-ci.com/broadinstitute/adapt.svg?token=cZz1u4yFrRiEZnJWzdho&branch=master)](https://travis-ci.com/broadinstitute/adapt) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/broadinstitute/adapt/pulls) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE) [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/adapt/README.html)
#### Activity-informed Design with All-inclusive Patrolling of Targets

ADAPT efficiently designs activity-informed nucleic acid diagnostics for viruses.

In particular, ADAPT designs assays with maximal predicted detection activity, in expectation over a virus's genomic diversity, subject to soft and hard constraints on the assay's complexity and specificity.
ADAPT's designs are:

* **Comprehensive**. Designs are effective against variable targets because ADAPT considers the full spectrum of their known genomic diversity.
* **Sensitive**. ADAPT leverages predictive models of detection activity.
It includes a pre-trained model of CRISPR-Cas13a detection activity, trained from ~19,000 guide-target pairs.
* **Specific**. Designs can distinguish related species or lineages within a species.
The approach accommodates G-U pairing, which is important in RNA applications.
* **End-to-end**. ADAPT automatically downloads and curates data from public databases to provide designs rapidly at scale.
The input can be as simple as a species or taxonomy in the form of an NCBI taxonomy identifier.

<br/>

ADAPT outputs a list of assay options ranked by predicted performance.
In addition to its objective that maximizes expected activity, ADAPT supports a simpler objective that minimizes the number of probes subject to detecting a specified fraction of diversity.

ADAPT includes a pre-trained model that predicts CRISPR-Cas13a guide detection activity, so ADAPT is directly suited to detection with Cas13a.
ADAPT's output also includes amplification primers, e.g., for use with the SHERLOCK platform.
The framework and software are compatible with other nucleic acid technologies given appropriate models.

For more information, see our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.11.28.401877v1) that describes ADAPT and evaluates its designs experimentally.

### Table of contents

* [Setting up ADAPT](#setting-up-adapt)
  * [Dependencies](#dependencies)
  * [Setting up a conda environment](#setting-up-a-conda-environment)
  * [Downloading and installing](#downloading-and-installing)
  * [Testing](#testing)
  * [Running on Docker](#running-on-docker)
* [Using ADAPT](#using-adapt)
  * [Overview](#overview)
  * [Required subcommands](#required-subcommands)
  * [Specifying the objective](#specifying-the-objective)
  * [Enforcing specificity](#enforcing-specificity)
  * [Searching for complete targets](#searching-for-complete-targets)
  * [Automatically downloading and curating data](#automatically-downloading-and-curating-data)
  * [Miscellaneous key arguments](#miscellaneous-key-arguments)
  * [Output](#output)
* [Examples](#examples)
  * [Basic: designing within sliding window](#basic-designing-within-sliding-window)
  * [Designing end-to-end with predictive model](#designing-end-to-end-with-predictive-model)
* [Support and contributing](#support-and-contributing)
  * [Questions](#questions)
  * [Contributing](#contributing)
  * [Citation](#citation)
  * [License](#license)
  * [Related repositories](#related-repositories)

<br/>

# Setting up ADAPT

## Dependencies

ADAPT requires:
* [Python](https://www.python.org) &gt;= 3.8.0
* [NumPy](http://www.numpy.org) &gt;= 1.16.0, &lt; 1.19.0
* [SciPy](https://www.scipy.org) == 1.4.1
* [TensorFlow](https://www.tensorflow.org) == 2.3.0

Using ADAPT with AWS cloud features additionally requires:
* [Boto3](https://aws.amazon.com/sdk-for-python/) &gt;= 1.14.54
* [Botocore](https://botocore.amazonaws.com/v1/documentation/api/latest/index.html) &gt;= 1.17.54

Installing ADAPT with `pip`, as described below, will install NumPy, SciPy, and TensorFlow if they are not already installed. Installing ADAPT with `pip` using the AWS cloud features, as described below, will install Boto3 and Botocore if they are not already installed as well.

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

Once you have conda, you can [create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) an environment for ADAPT with Python 3.8:
```bash
conda create -n adapt python=3.8
```
Then, you can activate the `adapt` environment:
```bash
conda activate adapt
```
After the environment is created and activated, you can install ADAPT as described below.
You will need to activate the environment each time you use ADAPT.

## Downloading and installing

ADAPT is available via [Bioconda](https://anaconda.org/bioconda/adapt) for GNU/Linux and Windows operating systems.
To install ADAPT via Bioconda, follow the instructions in [Setting up a conda environment](#setting-up-a-conda-environment) to install Miniconda and activate the environment, and then run the following command:
```bash
conda install -c bioconda adapt
```

On other operating systems (or if you wish to modify ADAPT's code), ADAPT can be installed by cloning the repository and installing the package with `pip`:
```bash
git clone git@github.com:broadinstitute/adapt.git
cd adapt
pip install -e .
```
Depending on your setup (i.e., if you do not have write permissions in the installation directory), you may need to supply `--user` to `pip install`.

If you want to be able to use AWS cloud features through ADAPT, replace the last line with the following:
```bash
pip install -e ".[AWS]"
```

## Testing

The package uses Python's `unittest` framework.
To execute all tests, run:
```bash
python -m unittest discover
```

## Running on Docker
_Note: This section is optional, but may be useful for more advanced users or developers. You will need to install [Docker](https://docs.docker.com/get-docker/)._

If you would like to run ADAPT using a Docker container rather than installing it, you may use one of our pre-built ADAPT images.

For ADAPT without cloud features, use the image ID `quay.io/broadinstitute/adapt`.

For ADAPT with cloud features, use the image ID `quay.io/broadinstitute/adaptcloud`.

To pull our Docker image to your computer, run:
```bash
docker pull [IMAGE-ID]
```

To run ADAPT on a Docker container, run:
```bash
docker run --rm [IMAGE-ID] "[COMMAND]"
```
To run with ADAPT memoizing to a local directory, run:
```bash
docker run --rm -v /path/to/memo/on/host:/memo [IMAGE-ID] "[COMMAND]"
```
To run the container interactively (opening a command line to the container), run:
```bash
docker run --rm -it [IMAGE-ID]
```

# Using ADAPT

## Overview

The main program for designing assays is [`design.py`](./bin/design.py).

Below, we refer to *guides* in reference to our pre-trained model for CRISPR-Cas13a guides and our testing of ADAPT's designs with Cas13a.
More generally, *guides* can be thought of as *probes* to encompass other diagnostic technologies.

[`design.py`](./bin/design.py) requires two subcommands:
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] ...
```

## Required subcommands

SEARCH-TYPE is one of:

* `complete-targets`: Search for the best assay options, each containing primer pairs and guides between them.
This is usually our recommended search type.
More information is in [Searching for complete targets](#searching-for-complete-targets).
(Example [here](#designing-end-to-end-with-predictive-model).)
* `sliding-window`: Search for guides within a sliding window of a fixed length, and output an optimal guide set for each window.
This is the much simpler search type and can be helpful when getting started.
(Example [here](#basic-designing-with-sliding-window).)

INPUT-TYPE is one of:

* `fasta`: The input is one or more FASTA files, each containing aligned sequences for a taxon.
If more than one file is provided, the search finds taxon-specific designs meant for differential identification of the taxa.
* `auto-from-args`: The input is a single NCBI taxonomy ID, and related information, provided as command-line arguments.
This fetches sequences for the taxon, then curates, clusters and aligns the sequences, and finally uses the generated alignment as input for design.
More information is in [Automatically downloading and curating data](#automatically-downloading-and-curating-data).
* `auto-from-file`: The input is a file containing a list of taxonomy IDs and related information.
This operates like `auto-from-args`, except ADAPT designs with specificity across the input taxa using a single index for evaluating specificity (as opposed to having to build it separately for each taxon).
More information is in [Automatically downloading and curating data](#automatically-downloading-and-curating-data).

### Positional arguments

The positional arguments &mdash; which specify required input to ADAPT &mdash; depend on the INPUT-TYPE.
These arguments are defined below for each INPUT-TYPE.

##### If INPUT-TYPE is `fasta`:

```bash
design.py [SEARCH-TYPE] fasta [fasta] [fasta ...] -o [out-tsv] [out-tsv ...]
```
where `[fasta]` is a path to an aligned FASTA file for a taxon and `[out-tsv]` specifies where to write the output TSV file.
If there are more than one space-separated FASTA, there must be an equivalent number of output TSV files; the _i_'th output gives designs for the _i_'th input FASTA.

##### If INPUT-TYPE is `auto-from-args`:

```bash
design.py [SEARCH-TYPE] auto-from-args [taxid] [segment] [out-tsv]
```
where `[taxid]` is an NCBI [taxonomy ID](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi), `[segment]` is a segment label (e.g., 'S') or 'None' if unsegmented, and `[out-tsv]` specifies where to write the output TSV file.

##### If INPUT-TYPE is `auto-from-file`:

```bash
design.py [SEARCH-TYPE] auto-from-file [in-tsv] [out-dir]
```
where `[in-tsv]` is a path to a file specifying the input taxonomies (run `design.py [SEARCH-TYPE] auto-from-file --help` for details) and `[out-dir]` specifies a directory in which to write the outputs.

### Details on all arguments

To see details on all the arguments available, run
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] --help
```
with the particular choice of subcommands substituted in for `[SEARCH-TYPE]` and `[INPUT-TYPE]`.

## Specifying the objective

ADAPT supports two objective functions, specified using the `--obj` argument:

* Maximize activity (`--obj maximize-activity`)
* Minimize complexity (`--obj minimize-guides`)

Details on each are below.

### Objective: maximizing activity

Setting `--obj maximize-activity` tells ADAPT to design sets of guides having maximal activity, in expectation over the input taxon's genomic diversity, subject to soft and hard constraints on the size of the guide set.
This is usually our recommended objective, especially with access to a predictive model.
With this objective, the following arguments to [`design.py`](./bin/design.py) are relevant:

* `-sgc SOFT_GUIDE_CONSTRAINT`: Soft constraint on the number of guides in a design option.
There is no penalty for a number of guides &le; SOFT_GUIDE_CONSTRAINT.
Having a number of guides beyond this is penalized linearly according to PENALTY_STRENGTH.
(Default: 1.)
* `-hgc HARD_GUIDE_CONSTRAINT`: Hard constraint on the number of guides in a design option.
The number of guides in a design option will always be &le; HARD_GUIDE_CONSTRAINT.
HARD_GUIDE_CONSTRAINT must be &ge; SOFT_GUIDE_CONSTRAINT.
(Default: 5.)
* `--penalty-strength PENALTY_STRENGTH`: Importance of the penalty when the number of guides exceeds the soft guide constraint.
For a guide set G, the penalty in the objective is PENALTY_STRENGTH\*max(0, |G| - SOFT_GUIDE_CONSTRAINT).
PENALTY_STRENGTH must be &ge; 0.
The value depends on the output values of the activity model and reflects a tolerance for more complexity in the assay; for the default pre-trained activity model included with ADAPT, reasonable values are in the range \[0.1, 0.5\].
(Default: 0.25.)
* `--maximization-algorithm [greedy|random-greedy]`: Algorithm to use for solving the submodular maximization problem.
'greedy' uses the canonical greedy algorithm (Nemhauser 1978) for constrained monotone submodular maximization, which can perform well in practice but has poor worst-case guarantees because the function is not monotone (unless PENALTY_STRENGTH is 0).
'random-greedy' uses a randomized greedy algorithm (Buchbinder 2014) for constrained non-monotone submodular maximization, which has good worst-case guarantees.
(Default: 'random-greedy'.)

Note that, when the objective is to maximize activity, this objective requires a predictive model of activity and thus `--predict-activity-model-path` should be specified (details in [Miscellaneous key arguments](#miscellaneous-key-arguments)).
If you wish to use this objective but cannot use our pre-trained Cas13a model nor another model, see the help message for the argument `--use-simple-binary-activity-prediction`.

### Objective: minimizing complexity

Setting `--obj minimize-guides` tells ADAPT to minimize the number of guides in an assay subject to constraints on coverage of the input taxon's genomic diversity.
With this objective, the following arguments to [`design.py`](./bin/design.py) are relevant:

* `-gm MISMATCHES`: Tolerate up to MISMATCHES mismatches when determining whether a guide detects a sequence.
This argument is mainly meant to be helpful in the absence of a predictive model of activity.
When using a predictive model of activity (via `--predict-activity-model-path` and `--predict-activity-thres`), this argument serves as an additional requirement for evaluating detection on top of the model; it can be effectively ignored by setting MISMATCHES to be sufficiently high.
(Default: 0.)
* `--predict-activity-thres THRES_C THRES_R`: Thresholds for determining whether a guide-target pair is active and highly active.
THRES_C is a decision threshold on the output of the classifier (in \[0,1\]); predictions above this threshold are decided to be active.
Higher values have higher precision and less recall.
THRES_R is a decision threshold on the output of the regression model (at least 0); predictions above this threshold are decided to be highly active.
Higher values limit the number of pairs determined to be highly active.
To count as detecting a target sequence, a guide must be: (i) within MISMATCHES mismatches of the target sequence; (ii) classified as active; and (iii) predicted to be highly active.
Using this argument requires also setting `--predict-activity-model-path` (see [Miscellaneous key arguments](#miscellaneous-key-arguments)).
As noted above, MISMATCHES can be set to be sufficiently high to effectively ignore `-gm`.
(Default: use the default thresholds included with the model.)
* `-gp COVER_FRAC`: Design guides such that at least a fraction COVER_FRAC of the genomes are detected by the guides.
(Default: 1.0.)
* `--cover-by-year-decay YEAR_TSV MIN_YEAR_WITH_COVG DECAY`: Group input sequences by year and set a distinct COVER_FRAC for each year.
See `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on this argument.
Note that when INPUT-TYPE is `auto-from-{file,args}`, this argument does not accept YEAR_TSV.

## Enforcing specificity

ADAPT can enforce strict specificity so that designs will distinguish related taxa.

When INPUT-TYPE is `auto-from-file`, ADAPT will automatically enforce specificity between the input taxa using a single specificity index.
ADAPT can also enforce specificity when designing for a single taxon by parsing the `--specific-against-*` arguments.

To enforce specificity, the following arguments to [`design.py`](./bin/design.py) are important:

* `--id-m ID_M` / `--id-frac ID_FRAC`: These parameters specify thresholds for determining specificity.
Allow for up to ID_M mismatches when determining whether a guide *hits a sequence* in a taxon other than the one for which it is being designed, and decide that a guide *hits a taxon* if it hits at least ID_FRAC of the sequences in that taxon.
ADAPT does not design guides that hit a taxon other than the one for which they are being designed.
Higher values of ID_M and lower values of ID_FRAC correspond to more strict specificity.
(Default: 4 for ID_M, 0.01 for ID_FRAC.)
* `--specific-against-fastas [fasta] [fasta ...]`: Design guides to be specific against the provided sequences (in FASTA format; do not need to be aligned).
That is, the guides should not hit sequences in these FASTA files, as measured by ID_M and ID_FRAC.
Each `[fasta]` is treated as a separate taxon when ID_FRAC is applied.
* `--specific-against-taxa SPECIFIC_TSV`: Design guides to be specific against the provided taxa.
SPECIFIC_TSV is a path to a TSV file where each row specifies a taxonomy with two columns: (1) NCBI taxonomy ID; (2) segment label, or 'None' if unsegmented.
That is, the guides should not hit sequences in these taxonomies, as measured by ID_M and ID_FRAC.

## Searching for complete targets

When SEARCH-TYPE is `complete-targets`, ADAPT performs a branch and bound search to find a collection of assay design options.
It finds the best _N_ design options for a specified _N_.
Each design option represents a genomic region containing primer pairs and guides between them.
There is no set length for the region.
The _N_ options are intended to be a diverse (non-overlapping) selection.

Below are key arguments to [`design.py`](./bin/design.py) when SEARCH-TYPE is `complete-targets`:

* `--best-n-targets BEST_N_TARGETS`: Only compute and output the best BEST_N_TARGETS design options, where each receives an objective value according to OBJ_FN_WEIGHTS.
Note that higher values of BEST_N_TARGETS can significantly increase runtime.
(Default: 10.)
* `--obj-fn-weights OBJ_FN_WEIGHTS`: Coefficients to use in an objective function for each design target.
See `design.py complete-targets [INPUT-TYPE] --help` for details.
* `-pl PRIMER_LENGTH`: Design primers to be PRIMER_LENGTH nt long.
(Default: 30.)
* `-pp PRIMER_COVER_FRAC`: Same as `-gp` described above, except for the design of primers.
(Default: 1.0.)
* `-pm PRIMER_MISMATCHES`: Tolerate up to PRIMER_MISMATCHES mismatches when determining whether a primer hybridizes to a sequence.
(Default: 0.)
* `--max-primers-at-site MAX_PRIMERS_AT_SITE`: Only allow up to MAX_PRIMERS_AT_SITE primers at each primer site.
If not set, there is no limit.
This argument is mostly intended to improve runtime &mdash; smaller values (~5) can significantly improve runtime on especially diverse viruses &mdash; because the number of primers is already penalized in the objective function.
Note that this is only an upper bound, and in practice the number will usually be less than it.
(Default: not set.)

## Automatically downloading and curating data

When INPUT-TYPE is `auto-from-{file,args}`, ADAPT will run end-to-end.
It fetches and curates genomes, clusters and aligns them, and uses the generated alignment as input for design.

Below are key arguments to [`design.py`](./bin/design.py) when SEARCH-TYPE is `auto-from-file` or `auto-from-args`:

* `--mafft-path MAFFT_PATH`: Use the [MAFFT](https://mafft.cbrc.jp/alignment/software/) executable at MAFFT_PATH for generating alignments.
* `--prep-memoize-dir PREP_MEMOIZE_DIR`: Memoize alignments and statistics on these alignments to the directory specified by PREP_MEMOIZE_DIR.
If repeatedly re-running on the same taxonomies, using this argument can significantly improve runtime across runs.
ADAPT can save the memoized information to an AWS S3 bucket by using the syntax `s3://BUCKET/PATH`, though this requires the AWS cloud installation mentioned in [Downloading and installing](#downloading-and-installing) and setting access key information.
Access key information can either be set using AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (details below) or by installing and configuring [AWS CLI](https://aws.amazon.com/cli/).
If not set (default), do not memoize information across runs.
* `--prep-influenza`: If set, use NCBI's influenza database for fetching data.
This should be specified if design is for influenza A/B/C viruses.
* `--sample-seqs SAMPLE_SEQS`: Randomly sample SAMPLE_SEQS accessions with replacement from each taxonomy, and move forward with the design using this sample.
This can be useful for measuring some properties of the design, or for faster runtime when debugging.
* `--cluster-threshold CLUSTER_THRESHOLD`: Use CLUSTER_THRESHOLD as the maximum inter-cluster distance when clustering sequences prior to alignment.
The distance is average nucleotide dissimilarity (1-ANI); higher values result in fewer clusters.
(Default: 0.2.)
* `--use-accessions USE_ACCESSIONS`: Use the specified NCBI GenBank accessions, in a file at the path USE_ACCESSIONS, for generating input.
ADAPT uses these accessions instead of fetching neighbors from NCBI, but it will still download the sequences for these accessions.
See `design.py [SEARCH-TYPE] auto-from-{file,args} --help` for details on the format of the file.

When using AWS S3 to memoize information across runs (`--prep-memoize-dir`), the following arguments are also important:

* `--aws-access-key-id AWS_ACCESS_KEY_ID` / `--aws-secret-access-key AWS_SECRET_ACCESS_KEY`: Use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to login to AWS cloud services.
Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are needed to login.
These arguments are only necessary if saving the memoized data to an S3 bucket using PREP_MEMOIZE_DIR and AWS CLI has not been installed and configured.
If AWS CLI has been installed and configured and these arguments are passed, they will override the AWS CLI configuration.

## Miscellaneous key arguments

In addition to the arguments above, there are others that are often important when running [`design.py`](./bin/design.py):

* `--predict-activity-model-path MODEL_C MODEL_R`: Modles that predict activity of guide-target pairs.
MODEL_C gives a classification model that predicts whether a guide-target pair is active, and MODEL_R gives a regression model that predicts a measure of activity on active pairs.
Each argument is a path to a serialized model in TensorFlow's SavedModel format.
Pre-trained classification and regression models are in [`models/`](./models).
With `--obj maximize-activity`, the models are essential because they inform ADAPT of the measurements it aims to maximize.
With `--obj minimize-guides`, the models constrain the design such that a guide must be highly active to detect a sequence (specified by `--predict-activity-thres`).
(Default: not set, which does not use predicted activity during design.)
* `-gl GUIDE_LENGTH`: Design guides to be GUIDE_LENGTH nt long.
(Default: 28.)
* `--do-not-allow-gu-pairing`: If set, do not count G-U (wobble) base pairs between guide and target sequence as matching.
By default, they count as matches.
This applies when `-gm` is used with `--obj minimize-guides` and when enforcing specificity.
* `--require-flanking5 REQUIRE_FLANKING5` / `--require-flanking3 REQUIRE_FLANKING3`: Require the given sequence on the 5' (REQUIRE_FLANKING5) and/or 3' (REQUIRE_FLANKING3) side of the protospacer for each designed guide.
This tolerates ambiguity in the sequence (e.g., 'H' requires 'A', 'C', or 'T').
This can enforce a desired protospacer flanking site (PFS) nucleotide; it can also accommodate multiple nucleotides (motif).
Note that this is the 5'/3' end in the target sequence (not the spacer sequence).
When a predictive model of activity is given, this argument is not needed; it can still be specified, however, as an additional requirement on top of how the model evaluates activity.

## Output

The files output by ADAPT are TSV files, but vary in format depending on SEARCH-TYPE and INPUT-TYPE.
There is a separate TSV file for each taxon.

For all cases, run `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` to see details on the output format and on how to specify paths to the output TSV files.

### Complete targets

When SEARCH-TYPE is `complete-targets`, each row gives an assay design option; there are BEST_N_TARGETS of them.
Each design option corresponds to a genomic region (amplicon).
The columns give the primer and guide sequences as well as additional information about them.
There are about 20 columns; some key ones are:
* `objective-value`: Objective value based on OBJ_FN_WEIGHTS.
* `target-start` / `target-end`: Start (inclusive) and end (exclusive) positions of the genomic region in the alignment generated by ADAPT.
* `{left,right}-primer-target-sequences`: Sequences of 5' and 3' primers, from the targets (see [Complementarity](#complementarity)).
Within each of the two columns (amplicon endpoints), if there are multiple sequences they are separated by spaces.
* `total-frac-bound-by-guides`: Fraction of all input sequences predicted to be detected by the guide set.
* `guide-set-expected-activity`: Predicted activity of the guide set in detecting the input sequences, in expectation over the input sequences.
(nan if no predictor is set.)
* `guide-set-median-activity` / `guide-set-5th-pctile-activity`: Median and 5th percentile of predicted activity of the guide set over the input sequences.
(nan if no predictor is set.)
* `guide-expected-activities`: Predicted activity of each separate guide in detecting the input sequences, in expectation over the input sequences.
They are separated by spaces; if there is only 1 guide, this is equivalent to `guide-set-expected-activity`.
(nan if no predictor is set.)
* `guide-target-sequences`: Sequences of guides, from the targets (see [Complementarity](#complementarity)!).
If there are multiple, they are separated by spaces.
* `guide-target-sequence-positions`: Positions of the guides in the alignment, in the same order as they are reported; a guide may come from >1 position, so positions are reported in set notation (e.g., \{100\}).

The rows in the output are sorted by the objective value: better options are on top.
Smaller values are better with `--obj minimize-guides` and larger values are better with `--obj maximize-activity`.

When INPUT-TYPE is `auto-from-file` or `auto-from-args` and ADAPT generates more than one cluster of input sequences, there is a separate TSV file for each cluster; the filenames end in `.0`, `.1`, etc.

### Sliding window

When SEARCH-TYPE is `sliding-window`, each row gives a window in the alignment and the columns give information about the guides designed for that window.
The columns are:
* `window-start` / `window-end`: Start (inclusive) and end (exclusive) positions in the alignment.
* `count`: Number of guide sequences.
* `score`: Statistic between 0 and 1 that describes the redundancy of the guides in detecting the input sequences (higher is better).
This is meant to break ties between windows with the same number of guide sequences, and is not intended to be compared between windows with different numbers of guides.
* `total-frac-bound`: Total fraction of all input sequences that are detected by a guide.
Note that if `--cover-by-year-decay` is provided, this might be less than COVER_FRAC.
* `target-sequences`: Sequences of guides, from the targets (see [Complementarity](#complementarity)!).
If there are multiple, they are separated by spaces.
* `target-sequence-positions`: Positions of the guides in the alignment, in the same order as they are reported; a guide may come from >1 position, so positions are reported in set notation (e.g., \{100\}).

By default, when SEARCH-TYPE is `sliding-window`, the rows in the output are sorted by the position of the window.
With the `--sort` argument to [`design.py`](./bin/design.py), ADAPT sorts the rows so that the "best" choices of windows are on top.
It sorts by `count` (ascending) followed by `score` (descending), so that windows with the fewest guides and highest score are on top.

### Complementarity

Note that output sequences are all in the same sense as the input (target) sequences.
**Synthesized guide sequences should be reverse complements of the output sequences!**
Likewise, synthesized primer sequences should account for this.

# Examples

## Basic: designing within sliding window

This is the most simple example.
**It does not download genomes, search for genomic regions to target, or use a predictive model of activity; for these features, see the next example.**

The repository includes an alignment of Lassa virus sequences (S segment) from Sierra Leone in `examples/SLE_S.aligned.fasta`.
Run:
```bash
design.py sliding-window fasta examples/SLE_S.aligned.fasta -o probes.tsv -w 200 -gl 28 -gm 1 -gp 0.95
```

From this alignment, ADAPT scans each 200 nt window (`-w 200`) to find the smallest collection of probes that:
* are all within the window
* are 28 nt long (`-gl 28`)
* detect 95% of all input sequences (`-gp 0.95`), tolerating up to 1 mismatch (`-gm 1`) between a probe and target

ADAPT outputs a file, `probes.tsv`, that contains the probe sequences for each window.
See [Output](#output) above for a description of this file.

## Designing end-to-end with predictive model

ADAPT can automatically download and curate sequences for its design, and search efficiently over possible genomic regions to find primers/amplicons as well as Cas13a guides.
It identifies Cas13a guides using a pre-trained predictive model of activity.

Run:
```bash
design.py complete-targets auto-from-args 64320 None guides.tsv -gl 28 --obj maximize-activity -pl 30 -pm 1 -pp 0.95 --predict-activity-model-path models/classify/model-51373185 models/regress/model-f8b6fd5d --best-n-targets 5 --mafft-path MAFFT_PATH --sample-seqs 50 --verbose
```
This downloads and designs assays to detect genomes of Zika virus (NCBI taxonomy ID [64320](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=64320)).
You must fill in `MAFFT_PATH` with an executable of MAFFT.

ADAPT designs primers and Cas13a guides within the amplicons, such that:
* guides have maximal predicted detection activity, in expectation over Zika's genomic diversity (`--obj maximize-activity`) 
* guides are 28 nt long (`-gl 28`) and primers are 30 nt long (`-pl 30`)
* primers capture 95% of sequence diversity (`-pp 0.95`), tolerating up to 1 mismatch for each (`-pm 1`)

ADAPT outputs a file, `guides.tsv.0`, that contains the best 5 design options (`--best-n-targets 5`) as measured by ADAPT's default objective function.
See [Output](#output) above for a description of this file.

This example randomly selects 50 sequences (`--sample-seqs 50`) prior to design to speed the runtime in this example; the command should take about 20 minutes to run in full.
Using `--verbose` provides detailed output and is usually recommended, but the output can be extensive.

Note that this example does not enforce specificity.

To find minimal guide sets, use `--obj minimize-guides` instead and set `-gm` and `-gp`.
With this objective, Cas13a guides are determined to detect a sequence if they (i) satisfy the number of mismatches specified with `-gm` and (ii) are predicted by the model to be highly active in detecting the sequence; `-gm` can be sufficiently high to rely entirely on the predictive model.
The output guides will detect a desired fraction of all genomes, as specified by `-gp`.

# Support and contributing

## Questions

If you have questions about ADAPT, please create an [issue](https://github.com/broadinstitute/adapt/issues).

## Contributing

We welcome contributions to ADAPT.
This can be in the form of an [issue](https://github.com/broadinstitute/adapt/issues) or [pull request](https://github.com/broadinstitute/adapt/pulls).

## Citation

ADAPT was started by Hayden Metsky, and is developed by Priya Pillai and Hayden.

If you find ADAPT useful to your work, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2020.11.28.401877v1) as:
  * Metsky HC _et al_. Efficient design of maximally active and specific nucleic acid diagnostics for thousands of viruses. _bioRxiv_ 2020.11.28.401877. doi:10.1101/2020.11.28.401877.

## License

ADAPT is licensed under the terms of the [MIT license](./LICENSE).

## Related repositories

There are other repositories on GitHub associated with ADAPT:
  * [adapt-seq-design](https://github.com/broadinstitute/adapt-seq-design): Predictive modeling library, datasets, training, and evaluation (applied to CRISPR-Cas13a).
  * [adapt-analysis](https://github.com/broadinstitute/adapt-analysis): Analysis of ADAPT's designs and benchmarking its computational performance, as well as miscellaneous analyses for the ADAPT paper.
  * [adapt-designs](https://github.com/broadinstitute/adapt-designs): Designs output by ADAPT, including all experimentally tested designs.
  * [adapt-pipes](https://github.com/broadinstitute/adapt-pipes): Workflows for running ADAPT on the cloud, tailored for AWS.
