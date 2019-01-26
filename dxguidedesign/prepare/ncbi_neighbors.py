"""Utilities for working with genome neighbors (complete genomes) from NCBI.
"""

import datetime
import gzip
import logging
import re
import tempfile
import time
import urllib.parse
import urllib.request

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def ncbi_neighbors_url(taxid):
    """Construct URL for downloading list of genome neighbors.

    Args:
        taxid: taxonomic ID to download neighbors for

    Returns:
        str representing download URL
    """
    params = urllib.parse.urlencode({'taxid': taxid, 'cmd': 'download2'})
    url = 'https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?%s' % params
    return url


def fetch_neighbors_table(taxid):
    """Fetch genome neighbors table from NCBI.

    Args:
        taxid: taxonomic ID to download neighbors for

    Yields:
        lines, where each line is from the genome neighbors
        table and each line is a str
    """
    logger.debug(("Fetching table of neighbors for tax %d") % taxid)

    url = ncbi_neighbors_url(taxid)
    r = urllib.request.urlopen(url)
    raw_data = r.read()
    for line in raw_data.decode('utf-8').split('\n'):
        line_rstrip = line.rstrip()
        if line_rstrip != '':
            yield line_rstrip


def ncbi_influenza_genomes_url():
    """Construct URL for downloading NCBI influenza genomes database.

    Returns:
        str representing download URL
    """
    url = 'ftp://ftp.ncbi.nih.gov/genomes/INFLUENZA/genomeset.dat.gz'
    return url


def fetch_influenza_genomes_table(species_name):
    """Fetch influenza genome table from NCBI.

    Args:
        species_name: filter to keep only lines that contain this species
            name

    Yields:
        lines, where each line is from the genome database table and
        each line is a str
    """
    logger.debug(("Fetching table of influenza genomes for species %s") %
        species_name)
    species_name_lower = species_name.lower()

    url = ncbi_influenza_genomes_url()
    r = urllib.request.urlopen(url)
    raw_data = gzip.GzipFile(fileobj=r).read()
    for line in raw_data.decode('utf-8').split('\n'):
        line_rstrip = line.rstrip()
        if line_rstrip != '':
            if species_name_lower in line_rstrip.lower():
                yield line_rstrip


def ncbi_download_url(accessions):
    """Construct URL for downloading FASTA sequence.

    Args:
        accessions: collection of accessions to download sequences for

    Returns:
        str representing download URL
    """
    ids = ','.join(accessions)
    # Use safe=',' to not encode ',' as '%2'
    params = urllib.parse.urlencode({'id': ids, 'db': 'nuccore',
        'rettype': 'fasta', 'retmode': 'text'}, safe=',')
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?%s' % params
    return url


def fetch_fastas(accessions, batch_size=100, reqs_per_sec=2):
    """Download sequences for accessions.

    Entrez enforces a limit of ~3 requests per second (or else it
    will return a 'Too many requests' error); to avoid this, this
    aims for ~2 requests per second. To use up to 10 requests per second,
    request an API key from Entrez.

    Args:
        accessions: collection of accessions to download sequences for
        batch_size: number of accessions to download in each batch
        reqs_per_sec: number of requests per second to allow

    Returns:
        tempfile object containing the sequences in fasta format
    """
    logger.debug(("Fetching fasta files for %d accessions") % len(accessions))

    # Make temp file
    fp = tempfile.NamedTemporaryFile()

    # Download sequences in batches
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:(i + batch_size)]
        url = ncbi_download_url(batch)
        r = urllib.request.urlopen(url)
        raw_data = r.read()
        for line in raw_data.decode('utf-8').split('\n'):
            fp.write((line + '\n').encode())
        time.sleep(1.0/reqs_per_sec)

    # Set position to 0 so it can be re-read
    fp.seek(0)

    return fp


class Neighbor:
    """Immutable representation of a genome neighbor."""

    def __init__(self, acc, refseq_acc, hosts, lineage, tax_name, segment,
            metadata={}):
        self.acc = acc
        self.refseq_acc = refseq_acc
        self.hosts = hosts
        self.lineage = lineage
        self.tax_name = tax_name
        self.segment = segment
        self.metadata = metadata

    def _list_of_attrs(self):
        """List of all attributes (except the accession)."""
        return [self.refseq_acc, self.hosts, self.lineage, self.tax_name,
            self.segment, self.metadata]

    def __eq__(self, other):
        return (self.acc == other.acc and
                self.refseq_acc == other.refseq_acc and
                sorted(self.hosts) == sorted(other.hosts) and
                self.lineage == other.lineage and
                self.tax_name == other.tax_name and
                self.segment == other.segment,
                self.metadata == other.metadata)

    def __repr__(self):
        return ';'.join('"' + str(s) + '"' for s in
            [self.acc] + self._list_of_attrs())

    def __str__(self):
        return self.acc + ' : ' + ', '.join('"' + str(s) + '"' for s in
            self._list_of_attrs())


def construct_neighbors(taxid):
    """Construct Neighbor objects for all neighbors of a taxonomic ID.

    Args:
        taxid: taxonomic ID to download neighbors for

    Returns:
        list of Neighbor objects
    """
    logger.info(("Constructing a list of neighbors for tax %d") % taxid)

    expected_col_order = ['Representative', 'Neighbor', 'Host',
        'Selected lineage', 'Taxonomy name', 'Segment name']

    neighbors = []
    for line in fetch_neighbors_table(taxid):
        if len(line.strip()) == 0:
            continue

        ls = line.split('\t')

        if line.startswith('##'):
            # Header line
            if line.startswith('## Columns:'):
                # Verify the columns are as expected
                col_names = [n.replace('"', '') for n in ls[1:]]
                if expected_col_order != col_names:
                    raise Exception(("The order of columns in the neighbor "
                        "list does not match the expected order"))
            # Skip the header
            continue

        refseq_acc = ls[0]
        acc = ls[1]
        hosts = ls[2].split(',')
        lineage = tuple(ls[3].split(','))
        tax_name = ls[4]
        segment = ls[5].replace('segment', '').strip()

        neighbor = Neighbor(acc, refseq_acc, hosts, lineage, tax_name, segment)
        neighbors += [neighbor]

    return neighbors


def construct_influenza_genome_neighbors(taxid):
    """Construct Neighbor objects for all influenza genomes

    According to the README on NCBI's influenza database FTP site:
    ```
    The genomeset.dat file contains information for sequences of viruses with a
    complete set of segments in full-length (or nearly
    full-length).  Those of the same virus are grouped together (using an internal
    group ID that is shown in the last column of the file) and separated by an
    empty line from those of other viruses.
    ```
    fetch_influenza_genomes_table() returns genomeset.dat filtered for
    a given species name.

    According to that same README, the columns are:
    ```
    GenBank accession number[tab]Host[tab]Genome segment number or protein name
    [tab]Subtype[tab]Country[tab]Year/month/date[tab]Sequence length
    [tab]Virus name[tab]Age[tab]Gender
    ```

    Args:
        taxid: taxonomic ID for an influenza species; must be influenza A
            or B species

    Returns:
        list of Neighbor objects
    """
    logger.info(("Constructing a list of neighbors for influenza species "
                 "with tax %d") % taxid)

    influenza_species = {11320: 'Influenza A virus',
                         11520: 'Influenza B virus'}
    if taxid not in influenza_species:
        raise ValueError(("Taxid (%d) must be for either influenza A or "
                          "influenza B virus species") % taxid)
    species_name = influenza_species[taxid]

    influenza_lineages = {11320: ('Orthomyxoviridae', 'Alphainfluenzavirus',
                                  'Influenza A virus'),
                          11520: ('Orthomyxoviridae', 'Betainfluenzavirus',
                                  'Influenza B virus')}
    lineage = influenza_lineages[taxid]

    # Construct a pattern to match years in a date (1000--2999)
    year_p = re.compile('([1-2][0-9]{3})')

    # Determine the current year
    curr_year = int(datetime.datetime.now().year)

    neighbors = []
    for line in fetch_influenza_genomes_table(species_name):
        if len(line.strip()) == 0:
            continue

        ls = line.split('\t')

        acc = ls[0]
        hosts = [ls[1]]
        segment = ls[2]
        subtype = ls[3]
        country = ls[4]
        date = ls[5]
        seq_len = int(ls[6])
        name = ls[7]

        # Parse the year
        year_m = year_p.match(date)
        if year_m is None:
            # No year available; skip the sequence
            continue
        year = int(year_m.group(1))
        if year > curr_year:
            # This year is in the future (probably a typo); skip the sequence
            continue

        # Construct dict of metadata
        metadata = {'subtype': subtype, 'country': country, 'year': year,
                    'seq_len': seq_len}

        # Leave the refseq_acc as None because, in the influenza database,
        # sequences are not assigned a particular RefSeq
        neighbor = Neighbor(acc, None, hosts, lineage, name, segment,
            metadata=metadata)
        neighbors += [neighbor]

    return neighbors
