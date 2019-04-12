"""Utilities for working with genome neighbors (complete genomes) from NCBI.
"""

from collections import defaultdict
import datetime
import gzip
import logging
import random
import re
import tempfile
import time
import urllib.parse
import urllib.request
from xml.dom import minidom

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def urlopen_with_tries(url, initial_wait=5, rand_wait_range=(1, 60),
        max_num_tries=5):
    """
    Open a URL via urllib with repeated tries.

    Often calling urllib.request.urlopen() fails with HTTPError, especially
    if there are multiple processes calling it. The reason is that NCBI
    has a cap on the number of requests per unit time, and the error raised
    is 'HTTP Error 429: Too Many Requests'.

    Args:
        url: url to open
        initial_wait: number of seconds to wait in between the first two
            requests; the wait for each subsequent request doubles in time
        rand_wait_range: tuple (a, b); in addition to waiting an amount of
            time that grows exponentially (starting with initial_wait), also
            wait a random number of seconds between a and b (inclusive).
            If multiple processes are started simultaneously, this helps to
            avoid them waiting on the same cycle
        max_num_tries: maximum number of requests to attempt to make

    Returns:
        result of urllib.request.urlopen()
    """
    num_tries = 0
    while num_tries < max_num_tries:
        try:
            num_tries += 1
            logger.debug(("Making request to open url: %s"), url)
            r = urllib.request.urlopen(url)
            return r
        except urllib.error.HTTPError:
            if num_tries == max_num_tries:
                # This was the last allowed try
                logger.warning(("Encountered HTTPError %d times (the maximum "
                    "allowed) when opening url: %s"), num_tries, url)
                raise
            else:
                # Pause for a bit and retry
                wait = initial_wait * 2**(num_tries - 1)
                rand_wait = random.randint(*rand_wait_range)
                total_wait = wait + rand_wait
                logger.info(("Encountered HTTPError when opening url; "
                    "sleeping for %d seconds, and then trying again"),
                    total_wait)
                time.sleep(total_wait)
        except:
            logger.warning(("Encountered unexpected error while opening "
                "url: %s"), url)
            raise


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
    r = urlopen_with_tries(url)
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
    r = urlopen_with_tries(url)
    raw_data = gzip.GzipFile(fileobj=r).read()
    for line in raw_data.decode('utf-8').split('\n'):
        line_rstrip = line.rstrip()
        if line_rstrip != '':
            if species_name_lower in line_rstrip.lower():
                yield line_rstrip


def ncbi_fasta_download_url(accessions):
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
        url = ncbi_fasta_download_url(batch)
        r = urlopen_with_tries(url)
        raw_data = r.read()
        for line in raw_data.decode('utf-8').split('\n'):
            fp.write((line + '\n').encode())
        time.sleep(1.0/reqs_per_sec)

    # Set position to 0 so it can be re-read
    fp.seek(0)

    return fp


def ncbi_xml_download_url(accessions):
    """Construct URL for downloading GenBank XML.

    Args:
        accessions: collection of accessions to download XML for

    Returns:
        str representing download URL
    """
    ids = ','.join(accessions)
    # Use safe=',' to not encode ',' as '%2'
    params = urllib.parse.urlencode({'id': ids, 'db': 'nuccore',
        'retmode': 'xml'}, safe=',')
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?%s' % params
    return url


def fetch_xml(accessions, batch_size=100, reqs_per_sec=2):
    """Download XML for accessions.

    Entrez enforces a limit of ~3 requests per second (or else it
    will return a 'Too many requests' error); to avoid this, this
    aims for ~2 requests per second. To use up to 10 requests per second,
    request an API key from Entrez.

    Args:
        accessions: collection of accessions to download XML for
        batch_size: number of accessions to download in each batch
        reqs_per_sec: number of requests per second to allow

    Returns:
        tempfile object containing the downloaded XML data
    """
    # Make temp file
    fp = tempfile.NamedTemporaryFile()

    # Only write the header once; otherwise, it will be written for each
    # beach, and then the file will not be able to be parsed
    def is_xml_header(line):
        return (line.startswith('<?xml ') or line.startswith('<!DOCTYPE '))

    # Download in batches
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:(i + batch_size)]
        url = ncbi_xml_download_url(batch)
        r = urlopen_with_tries(url)
        raw_data = r.read()
        for line in raw_data.decode('utf-8').split('\n'):
            if i > 0 and is_xml_header(line):
                # Only write XML header for the first batch (i == 0)
                continue
            if i > 0 and '<GBSet>' in line:
                # Do not write GBSet open after the first batch
                line = line.replace('<GBSet>', '')
            if '</GBSet>' in line:
                # Never write GBSet close until the end
                line = line.replace('</GBSet>', '')
            fp.write((line + '\n').encode())
        time.sleep(1.0/reqs_per_sec)

    # Write the GBSet close
    fp.write(('</GBSet>' + '\n').encode())

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
    encountered_header = False
    for line in fetch_neighbors_table(taxid):
        if len(line.strip()) == 0:
            continue

        ls = line.split('\t')

        if line.startswith('##'):
            # Header line
            encountered_header = True
            if line.startswith('## Columns:'):
                # Verify the columns are as expected
                col_names = [n.replace('"', '') for n in ls[1:]]
                if expected_col_order != col_names:
                    raise Exception(("The order of columns in the neighbor "
                        "list does not match the expected order"))
            # Skip the header
            continue

        if not encountered_header:
            logger.critical(("Neighbors table for tax %d did not contain "
                "the expected header; it is possible that this is not a "
                "valid taxonomy ID"), taxid)

        refseq_acc = ls[0]
        acc = ls[1]
        hosts = ls[2].split(',')
        lineage = tuple(ls[3].split(','))
        tax_name = ls[4]
        segment = ls[5].replace('segment', '').strip()

        neighbor = Neighbor(acc, refseq_acc, hosts, lineage, tax_name, segment)
        neighbors += [neighbor]

    return neighbors


def add_metadata_to_neighbors_and_filter(neighbors):
    """Fetch and add metadata to neighbors.

    This only fetches for neighbors that do not have metadata set.

    This also filters out neighbors without a known year.

    Args:
        neighbors: collection of Neighbor objects

    Returns:
        neighbors, with metadata included (excluding the ones filtered out)
    """
    # Fetch metadata for each neighbor without metadata
    to_fetch = set(n.acc for n in neighbors if n.metadata == {})
    if len(to_fetch) > 0:
        metadata = fetch_metadata(to_fetch)
    else:
        metadata = {}
    acc_to_skip = set()
    for neighbor in neighbors:
        if neighbor.acc in to_fetch:
            neighbor.metadata = metadata[neighbor.acc]
        if neighbor.metadata['year'] is None:
            acc_to_skip.add(neighbor.acc)

    # Requiring year, so remove accessions that do not have a year
    if len(acc_to_skip) > 0:
        logger.warning(("Leaving out %d accessions that do not contain "
            "a year"), len(acc_to_skip))
    neighbors = [n for n in neighbors if n.acc not in acc_to_skip]

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
        year_m = year_p.search(date)
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


def parse_genbank_xml_for_source_features(fn):
    """Parse GenBank XML to extract source features.

    Args:
        fn: path to XML file, as generated by GenBank

    Returns:
        dict {accession: [(qualifier name, qualifier value)]}
    """
    doc = minidom.parse(fn)
    def parse_xml_node_value(element, tag_name):
        return element.getElementsByTagName(tag_name)[0].firstChild.nodeValue

    source_features = defaultdict(list)

    seqs = doc.getElementsByTagName('GBSeq')
    for seq in seqs:
        accession = parse_xml_node_value(seq, 'GBSeq_primary-accession')
        feature_table = seq.getElementsByTagName('GBSeq_feature-table')[0]
        for feature in feature_table.getElementsByTagName('GBFeature'):
            feature_key = parse_xml_node_value(feature, 'GBFeature_key')
            if feature_key == 'source':
                quals = feature.getElementsByTagName('GBFeature_quals')[0]
                for qualifier in quals.getElementsByTagName('GBQualifier'):
                    qual_name = parse_xml_node_value(qualifier, 'GBQualifier_name')
                    qual_value = parse_xml_node_value(qualifier, 'GBQualifier_value')
                    source_features[accession].append((qual_name, qual_value))

    return source_features


def fetch_metadata(accessions):
    """Fetch metadata from GenBank for accessions.

    This currently only parses out country and collection year.

    Args:
        accessions: collection of accessions to fetch for

    Returns:
        dict {accession: {'country': country, 'year': collection-year}}
    """
    accessions = list(set(accessions))
    logger.info(("Fetching metadata for %d accessions"), len(accessions))

    # Fetch XML and parse it for source features
    xml_tf = fetch_xml(accessions)
    source_features = parse_genbank_xml_for_source_features(xml_tf.name)

    # Construct a pattern to match years in a date (1000--2999)
    year_p = re.compile('([1-2][0-9]{3})')

    # Determine the current year
    curr_year = int(datetime.datetime.now().year)

    metadata = {}
    for accession, feats in source_features.items():
        year = None
        country = None
        for (name, value) in feats:
            if name == 'collection_date':
                # Parse the year
                year_m = year_p.search(value)
                if year_m is None:
                    # No year available
                    feat_year = None
                else:
                    feat_year = int(year_m.group(1))
                    if feat_year > curr_year:
                        # This year is in the future (probably a typo)
                        # Treat it as unavailable
                        feat_year = None
                    if year is not None and feat_year != year:
                        raise Exception(("Inconsistent year for "
                            "accession %s") % accession)
                year = feat_year
            if name == 'country':
                if country is not None and value != country:
                    raise Exception(("Inconsistent country for "
                        "accession %s") % accession)
                country = value
        metadata[accession] = {'country': country, 'year': year}

    # Close the tempfile
    xml_tf.close()

    return metadata

