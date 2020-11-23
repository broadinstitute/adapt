# ADAPT &nbsp;&middot;&nbsp; [![Build Status](https://travis-ci.com/broadinstitute/adapt.svg?token=cZz1u4yFrRiEZnJWzdho&branch=master)](https://travis-ci.com/broadinstitute/adapt) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/broadinstitute/adapt/pulls) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
#### Activity-informed Design with All-inclusive Patrolling of Targets

ADAPT is a software package for designing sensitive and specific nucleic acid viral diagnostics.

* **Comprehensive**: ADAPT considers the full spectrum of known genomic diversity for targeted taxa, so designs are effective against variable targets. This is critical for many viral species.
* **Sensitive**: ADAPT accepts predictive models of detection activity, so designs are predicted to be highly active against targets. It includes a pre-trained model of CRISPR-Cas13a detection activity, trained from ~19,000 guide-target pairs.
* **Specific**: ADAPT enforces strict specificity, so designs can distinguish related species or lineages within a species. The approach accommodates G-U pairing, which greatly increases the chance of off-target hits in some applications.
* **End-to-end**: ADAPT automatically downloads and curates data from genome databases, so it provides designs rapidly at scale. The input can be as simple as a species or other taxonomy in the form of an NCBI taxonomy identifier.

<br/>

ADAPT's main objective is to design assays that maximize predicted detection activity, in expectation over a taxon's genomic diversity, subject to soft and hard constraints on the assay's complexity and specificity.
The output is a list of design options, ranked according to anticipated performance.
ADAPT also supports a simpler objective function that minimizes a number of probes subject to detecting a specified fraction of known diversity.

The software package includes a pre-trained model of CRISPR-Cas13a detection activity, and therefore it is directly suited to detection with Cas13a.
ADAPT's output includes amplification primers, e.g., for use with the SHERLOCK platform.
However, the framework and software are compatible with other diagnostic technologies given appropriate models.

For more information, see the bioRxiv preprint describing and evaluating ADAPT.

<br/>

### Table of contents

* [Setting up ADAPT](#setting-up-adapt)
  * [Dependencies](#dependencies)
  * [Setting up a conda environment](#setting-up-a-conda-environment)
  * [Downloading and installing](#downloading-and-installing)
  * [Testing](#testing)
  * [Running on Docker](#running-on-docker)
* [Using ADAPT](#using-adapt)
  * [Designing assays](#designing-assays)
  * [Objective](#objective)
  * [Common options](#common-options)
  * [Output](#output)
* [Examples](#examples)
  * [Basic: designing with sliding window](#basic-designing-with-sliding-window)
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
* [Python](https://www.python.org) &gt;= 3.8
* [NumPy](http://www.numpy.org) &gt;= 1.16.0, &lt; 1.19.0
* [SciPy](https://www.scipy.org) == 1.4.1
* [TensorFlow](https://www.tensorflow.org) &gt;= 2.3.0

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

An easy way to setup ADAPT is to clone the repository and install the package with `pip`:
```bash
git clone git@github.com:broadinstitute/adapt.git
cd adapt
pip install -e .
```
Depending on your setup (i.e., if you do not have write permissions in the installation directory), you may need to supply `--user` to `pip install`.

If you want to be able to use AWS Cloud features, replace the last line with the following:
```bash
pip install -e ".[AWS]"
```

## Testing

The package uses Python's `unittest` framework.
To execute all unit tests, run:
```bash
python -m unittest discover
```

## Running on Docker
_Note: This section is optional, but may be useful for more advanced users or developers. You will need to install [Docker Desktop](https://docs.docker.com/get-docker/) to run on Docker._

If you would like to run ADAPT using a Docker container, you may use one of our pre-built images (not yet released; coming soon) or build your own locally.

### Getting a Docker Image

To build a Docker image locally with ADAPT installed, run:
```bash
docker build . -t adapt
```
If you would like to build a Docker image locally with ADAPT's AWS Cloud features, also run:
```bash
docker build . -t adaptcloud -f ./cloud.Dockerfile
```

If you would like to use one of our pre-built images, run:
```bash
docker pull [REGISTRY-PATH]
```
where `[REGISTRY-PATH]` is the URL to the image excluding `https://`.

### Running in a Docker Container

To run ADAPT on a Docker container, run:
```bash
docker run --rm [IMAGE-ID] "[COMMAND]"
```
To run with a local directory serving as the memo, run:
```bash
docker run --rm -v /path/to/memo/on/host:/memo [IMAGE-ID] "[COMMAND]"
```
To run the container interactively (opening a command line to the container), run:
```bash
docker run --rm -it [IMAGE-ID]
```

If you built a Docker image locally, `[IMAGE-ID]` is `adapt`/`adaptcloud`. If you are using one of our pre-built images, `[IMAGE-ID]` is the same as the `[REGISTRY-PATH]`.

# Using ADAPT

## Overview

The main program for designing assays is [`design.py`](./bin/design.py).

Below, we refer to *guides* in reference to our pre-trained model and testing for CRISPR-Cas13a guides.
However, more generally, they can be thought of as *probes* to encompass other diagnostic technologies.

[`design.py`](./bin/design.py) requires two subcommands:
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] ...
```

## Required subcommands

SEARCH-TYPE is one of:

* `complete-targets`: Search the best design options, each with primer pairs and guides between them.
This is usually our recommended search type.
More information is in [Searching for complete targets](#searching-for-complete-targets).
(Example [here](#designing-end-to-end-with-predictive-model).)
* `sliding-window`: Search for guides within a sliding window of a fixed length, and output an optimal guide set for each window.
This is the much simpler search type and can be helpful in getting started.
(Example [here](#basic-designing-with-sliding-window).)

INPUT-TYPE is one of:

* `fasta`: The input is one or more FASTA files, each containing aligned sequences for a taxon.
If more than one file is provided, the search performs differential identification across the taxa.
* `auto-from-args`: The input is a single taxonomic ID, and related information, provided as command-line arguments.
This fetches sequences for the taxon, then curates, clusters and aligns the sequences, and finally uses the generated alignment as input for design.
More information is in [Automatically downloading and curating data](#automatically-downloading-and-curating-data).
* `auto-from-file`: The input is a file containing a list of taxon IDs and related information.
This operates like `auto-from-args`, except ADAPT designs with specificity across the input taxa with a single index for evaluating specificity (as opposed to having to build it separately for each taxon).
More information is [Automatically downloading and curating data](#automatically-downloading-and-curating-data).

To see details on all the arguments available for a particular choice of subcommands, run:
```bash
design.py [SEARCH-TYPE] [INPUT-TYPE] --help
```
These details may include required positional arguments for a choice of subcommands.

## Specifying the objective

ADAPT supports two objective functions, specified using the `--obj` argument, to identify guide sets:

* Maximize activity (`--obj maximize-activity`)
* Minimize complexity (`--obj minimize-guides`)

Details on each are below.

### Objective: maximizing activity

Setting `--obj maximize-activity` tells ADAPT to design guide designs with maximal activity, in expectation over the input taxon's genomic diversity, subject to soft and hard constraints on the size of the guide set.
This is usually our recommended objective, especially with access to a predictive model.
With this objective, the following arguments to [`design.py`](./bin/design.py) are relevant:

* `-sgc SOFT_GUIDE_CONSTRAINT`: Soft constraint on the number of guides.
There is no penalty for a number of guides &le; SOFT_GUIDE_CONSTRAINT.
Having a number of guides beyond this is penalized linearly according to PENALTY_STRENGTH.
(Default: 1.)
* `-hgc HARD_GUIDE_CONSTRAINT`: Hard constraint on the number of guides.
The number of guides in a design will be &le; HARD_GUIDE_CONSTRAINT.
HARD_GUIDE_CONSTRAINT must be &ge; SOFT_GUIDE_CONSTRAINT.
(Default: 5.)
* `--penalty-strength PENALTY_STRENGTH`: Importance of the penalty when the number of guides exceeds the soft guide constraint.
For a guide set G, the penalty in the objective is PENALTY_STRENGTH\*max(0, |G| - SOFT_GUIDE_CONSTRAINT).
Must be &ge; 0.
The value depends on the output values of the activity model and reflects a tolerance for more guides; for the default activity model, reasonable values are in the range \[0.1, 0.5\].
(Default: 0.25.)
* `--maximization-algorithm [greedy|random-greedy]`: Algorithm to use for solving the submodular maximization problem.
'greedy' uses the canonical greedy algorithm (Nemhauser 1978) for constrained monotone submodular maximization, which can perform well in practice but has poor worst-case guarantees because the function is not monotone (unless PENALTY_STRENGTH is 0).
'random-greedy' uses a randomized greedy algorithm (Buchbinder 2014) for constrained non-monotone submodular maximization, which has good worst-case guarantees.
(Default: 'random-greedy'.)

Note that, when the objective is to maximize activity, this requires a predictive model of activity and thus `--predict-activity-model-path` (described below) must be specified.

### Objective: minimizing complexity

Setting `--obj minimize-guides` tells ADAPT to minimize the number of guides in an assay subject to constraints on coverage of the input taxon's genomic diversity.
With this objective, the following arguments to [`design.py`](./bin/design.py) are relevant:

* `-gm MISMATCHES`: Tolerate up to MISMATCHES mismatches when determining whether a guide hybridizes to a sequence.
(Default: 0.)
* `-gp COVER_FRAC`: Design guides such that at least a fraction COVER_FRAC of the genomes are hit by the guides.
(Default: 1.0.)
* `--cover-by-year-decay YEAR_TSV MIN_YEAR_WITH_COVG DECAY`: Group input sequences by year and set a separate desired COVER_FRAC for each year.
See `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on this argument.
Note that when INPUT-TYPE is `auto-from-{file,args}`, this argument does not accept YEAR_TSV.
* `--predict-activity-thres THRES_C THRES_R`: Thresholds for determining whether a guide-target pair is active and highly active.
THRES_C is a decision threshold on the output of the classifier (in \[0,1\]); predictions above this threshold are decided to be active.
Higher values have higher precision and less recall.
THRES_R is a decision threshold on the output of the regression model (at least 0); predictions above this threshold are decided to be highly active.
Higher values limit the number of pairs determined to be highly active.
When this argument is set, to count as detecting a target sequence, a guide must be: (a) within MISMATCHES mismatches of the target sequences; (b) classified as active; and (c) determined to be highly active against the target sequence.
This argument requires also setting `--predict-activity-model-path`.
(Default: use the default thresholds included with the model.)

## Enforcing specificity

ADAPT can enforce strict specificity so that designs can distinguish related taxa.

When INPUT-TYPE is `auto-from-file`, ADAPT will automatically enforce specificity between the input taxa with a single index.
ADAPT can also enforce specificity when designing for a single taxon with the `--specific-against-*` arguments.

To enforce specificity, the following arguments to [`design.py`](./bin/design.py) are important:

* `--id-m ID_M` / `--id-frac ID_FRAC`: Design guides to perform differential identification where these parameters determine specificity.
Allow for up to ID_M mismatches when determining whether a guide hits a sequence in a taxon other than the one for which it is being designed, and decide that a guide hits a taxon if it hits at least ID_FRAC of the sequences in that taxon.
ADAPT does not output guides that hit group/taxons other than the one for which they are being designed.
Higher values of ID_M and lower values of ID_FRAC correspond to more specificity.
(Default: 4 for ID_M, 0.01 for ID_FRAC.)
* `--specific-against-fastas [fasta] [fasta ...]`: Design guides to be specific against the provided sequences (in FASTA format; do not need to be aligned).
That is, the guides should not hit sequences in these FASTA files, as measured by ID_M and ID_FRAC.
* `--specific-against-taxa SPECIFIC_TSV`: Design guides to be specific against the provided taxa.
SPECIFIC_TSV is a path to a TSV file where each row specifies a taxonomy with two columns: (1) NCBI taxonomic ID; (2) segment label, or 'None' if unsegmented.
That is, the guides should not hit sequences in these taxonomies, as measured by ID_M and ID_FRAC.

## Searching for complete targets

When SEARCH-TYPE is `complete-targets`, ADAPT performs a brand and bound search to find a diverse collection of design options.
It finds the best _N_ design options for a specified _N_
Each design option represents a genomic region with primer pairs and guides between them.
There is no set length for the region.
The _N_ options are intended to be a diverse (non-overlapping) selection.

Below are key arguments to [`design.py`](./bin/design.py) when SEARCH-TYPE is `complete-targets`.

* `--best-n-targets BEST_N_TARGETS`: Only compute and output the best BEST_N_TARGETS design options, where receives an objective value according to OBJ_FN_WEIGHTS.
Note that higher values can significantly increase runtime.
(Default: 10.)
* `--obj-fn-weights OBJ_FN_WEIGHTS`: Coefficients to use in an objective function for each design target.
See `design.py complete-targets [INPUT-TYPE] --help` for details.
* `-pl PRIMER_LENGTH`: Design primers to be PRIMER_LENGTH nt long.
(Default: 30.)
* `-pp PRIMER_COVER_FRAC`: Same as `-gp` described above, except for the design of primers.
(Default: 1.0.)
* `-pm PRIMER_MISMATCHES`: Tolerate up to PRIMER_MISMATCHES mismatches when determining whether a primer hybridizes to a sequence.
(Default: 0.)
* `--max-primers-at-site MAX_PRIMERS_AT_SITE`: Only allow up to MAX_PRIMERS_AT_SITE primers at each primer set.
If not set, there is no limit.
Smaller values can significantly improve runtime.
(Default: not set.)

## Automatically downloading and curating data

When INPUT-TYPE is `auto-from-{file,args}`, ADAPT will run end-to-end.
It fetches and curates genomes, clusters and aligns them, and finally uses the generated alignment as input for design.

Below are key arguments to [`design.py`](./bin/design.py) when SEARCH-TYPE is `auto-from-file` or `auto-from-args`.

* `--mafft-path MAFFT_PATH`: Use the [MAFFT](https://mafft.cbrc.jp/alignment/software/) executable at MAFFT_PATH for generating alignments.
* `--prep-memoize-dir PREP_MEMOIZE_DIR`: Memoize alignments and statistics on these alignments in PREP_MEMOIZE_DIR.
This can save the memoized information to an S3 bucket by using the syntax `s3://BUCKET/PATH`, though this requires the AWS cloud installation mentioned in [Downloading and installing](#downloading-and-installing) and setting access key information.
Access key information can either be set using AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or by installing and configuring [AWS CLI](https://aws.amazon.com/cli/).
If repeatedly re-running on the same taxonomies, using this argument can significantly improve runtime.
If not set (default), do not memoize information across runns.
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

When using AWS S3 to memoize information across runs (`--prep-memoize-dir`), the following arguments are also important:

* `--aws-access-key-id AWS_ACCESS_KEY_ID`: Use AWS_ACCESS_KEY_ID to log in to AWS cloud services.
Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are necessary to log in.
This is only necessary if saving the memo to an S3 bucket using PREP_MEMOIZE_DIR and AWS CLI has not been installed and configured.
If AWS CLI has been installed and configured and this argument is passed in, AWS_ACCESS_KEY_ID will override the AWS CLI configuration.
* `--aws-secret-access-key AWS_SECRET_ACCESS_KEY`: Use AWS_SECRET_ACCESS_KEY to log in to AWS cloud services.
Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are necessary to log in.
This is only necessary if saving the memo to an S3 bucket using PREP_MEMOIZE_DIR and AWS CLI has not been installed and configured.
If AWS CLI has been installed and configured and this argument is passed in, AWS_ACCESS_KEY_ID will override the AWS CLI configuration.

## Miscellaneous key arguments

In addition to the arguments above, there are others that are often important when running [`design.py`](./bin/design.py):

* `--predict-activity-model-path MODEL_C MODEL_R`: Predict activity of guide-target pairs and only count guides as detecting a target if they are predicted to be highly active against it.
MODEL_C is for a classification model that predicts whether a guide-target pair is active, and MODEL_R is for a regression model that predicts a measure of activity on active pairs.
Each argument is a path to a serialized model in TensorFlow's SavedModel format.
Example classification and regression models are in [`models/`](./models).
(Default: not set, which does not use predicted activity as a constraint during design.)
* `-gl GUIDE_LENGTH`: Design guides to be GUIDE_LENGTH nt long.
(Default: 28.)
* `--do-not-allow-gu-pairing`: If set, do not count G-U (wobble) base pairs between guide and target sequence as matching.
* `--require-flanking5 REQUIRE_FLANKING5` / `--require-flanking3 REQUIRE_FLANKING3`: Require the given sequence on the 5' (REQUIRE_FLANKING5) and/or 3' (REQUIRE_FLANKING3) side of the protospacer for each designed guide.
This tolerates ambiguity in the sequence (e.g., 'H' requires 'A', 'C', or 'T').
This can enforce a desired protospacer flanking site (PFS) nucleotide; it can also accommodate multiple nucleotides (motif).
Note that this is the 5'/3' end in the target sequence (not the spacer sequence).
When a predictive model of activity is given, this argument is not needed; it can still be specified, however, as an additional requirement on top of how the model evaluates activity.

## Output

The files output by ADAPT are TSV files, but vary in format depending on SEARCH-TYPE and INPUT-TYPE.
There is a separate TSV file for each taxon.

For all cases, see `design.py [SEARCH-TYPE] [INPUT-TYPE] --help` for details on the output format and how to specify paths to the output TSV files.

### Complete targets

When SEARCH-TYPE is `complete-targets`, each row is a possible design option (primer pair and guide combination) and there are additional columns giving information about primer pairs and the guide sets.
There is also an `objective-value` column, giving the objective value of each design option according to `--obj-fn-weights`.
The rows in the output are sorted by the objective value (better options are on top); smaller values are better with `--obj minimize-guides` and larger values are better with `--obj maximize-activity`.

When INPUT-TYPE is `auto-from-file` or `auto-from-args`, there is a separate TSV file for each cluster of input sequences.

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

### Complementarity

Note that output sequences are in the same sense as the input sequences.
**Synthesized guide sequences should be reverse complements of the output!**
Likewise, synthesized primer sequences should account for this.

# Examples

## Basic: designing with sliding window

This is the most simple example.
It does not download genomes, search for genomic regions to target, or use a predictive model; for these features, see the next example.

The repository includes an alignment of Lassa virus sequences (S segment) from Sierra Leone.
Run:
```bash
design.py sliding-window fasta examples/SLE_S.aligned.fasta -o probes.tsv -w 200 -gl 28 -gm 1 -gp 0.95
```
This uses the alignment in `examples/SLE_S.aligned.fasta`.

From this alignment, it scans each 200 nt window (`-w 200`) to find the smallest collection of probes that:
* are all within the window
* are 28 nt long (`-gl 28`)
* detect 95% of all input sequences (`-gp 0.95`), tolerating up to 1 mismatch (`-gm 1`)

It outputs a file, `probes.tsv`, that contains the probe sequences for each window.
See [Output](#output) above for a description of this file.

## Designing end-to-end with predictive model

ADAPT can automatically download and curate sequences for its design, and search efficiently over possible genomic regions to find primers/amplicons as well as Cas13a guides.
It identifies Cas13a guides using the pre-trained predictive model.

Run:
```bash
design.py complete-targets auto-from-args 64320 None NC_035889 guides.tsv -gl 28 --obj maximize-activity -pl 30 -pm 1 -pp 0.95 --predict-activity-model-path models/classify/model-51373185 models/regress/model-f8b6fd5d --best-n-targets 5 --mafft-path MAFFT_PATH --sample-seqs 50 --verbose
```
This downloads and designs against genomes of Zika virus (NCBI taxonomy ID [64320](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=64320)).
You must fill in `MAFFT_PATH` with an executable.

This designs primers and Cas13a guides within the amplicons they detect, such that:
* guides are 28 nt long (`-gl 28`) and primers are 30 nt long (`-pl 30`)
* guides have maximal predicted detection activity, in expectation over genomic diversity (`--obj maximize-activity`) 
* primers capture 95% of sequence diversity (`-pp 0.95`), tolerating up to 1 mismatch for each (`-pm 1`)

It outputs a file, `guides.tsv.0`, that contains the best 5 design choices (`--best-n-targets 5`) as measured by the objective function.
See [Output](#output) above for a description of this file.

This example randomly selects 50 sequences (`--sample-seqs 50`) prior to design to speed the runtime in this example; the command should take about 20 minutes to run in full.
Using `--verbose` provides detailed output and is usually recommended, but the output can be extensive.

Note that this example does not account for taxon-specificity.

To find minimal guide sets, use `--obj minimize-guides` instead and set `-gm` and `-gp`.
With this objective, Cas13a guides are determined to detect a sequence if they (i) satisfy the number of mismatches specified with `-gm`; and (ii) are predicted by the model to be highly active in detecting the sequence; `-gm` can be sufficiently high to rely entirely on the predictive model.
The output guides will detect a desired fraction of all genomes, as specified by `-gp`.

# Support and contributing

## Questions

If you have questions about ADAPT, please create an [issue](https://github.com/broadinstitute/adapt/issues).

## Contributing

We welcome contributions to ADAPT.
This can be in the form of an [issue](https://github.com/broadinstitute/adapt/issues) or [pull request](https://github.com/broadinstitute/adapt/pulls).

## Citation

ADAPT was started by Hayden Metsky, and is developed by Priya Pillai and Hayden.

If you find ADAPT useful to your work, please cite our preprint as:
  * **[CITATION HERE]**

## License

ADAPT is licensed under the terms of the [MIT license](./LICENSE).

## Related repositories

There are other repositories on GitHub associated with ADAPT:
  * [adapt-seq-design](https://github.com/broadinstitute/adapt-seq-design): Predictive modeling library, datasets, training, and evaluation (applied to CRISPR-Cas13a).
  * [adapt-analysis](https://github.com/broadinstitute/adapt-analysis): Analysis of ADAPT's designs and benchmarking its computational performance, as well as miscellaneous analyses for the ADAPT paper.
  * [adapt-designs](https://github.com/broadinstitute/adapt-designs): Designs output by ADAPT, including all experimentally tested designs.
  * [adapt-pipes](https://github.com/broadinstitute/adapt-pipes): Workflows for running ADAPT on the cloud, tailored for AWS.
