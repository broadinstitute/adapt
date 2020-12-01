"""Utilities for working with genome neighbors (complete genomes) from NCBI.
"""

from collections import defaultdict
import datetime
import gzip
import http.client
import logging
import random
import re
import socket
import tempfile
import time
import urllib.parse
import urllib.request
from xml.dom import minidom
from os import unlink

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


# Global variable for API key
ncbi_api_key = None


def urlopen_with_tries(url, initial_wait=5, rand_wait_range=(1, 60),
        max_num_tries=10, timeout=60, read=False):
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
        timeout: timeout in sec before retrying
        read: also try to read the opened URL, and return the results;
            if this raises an HTTPException, the call will be retried

    Returns:
        result of urllib.request.urlopen(); unless read is True, in which
        case it is the data returned by reading the url
    """
    num_tries = 0
    while num_tries < max_num_tries:
        try:
            num_tries += 1
            logger.debug(("Making request to open url: %s"), url)
            r = urllib.request.urlopen(url, timeout=timeout)
            if read:
                raw_data = r.read()
                return raw_data
            else:
                return r
        except (urllib.error.HTTPError, http.client.HTTPException,
                urllib.error.URLError, socket.timeout):
            if num_tries == max_num_tries:
                # This was the last allowed try
                logger.warning(("Encountered HTTPError or HTTPException or "
                    "URLError or timeout %d times (the maximum allowed) when "
                    "opening url: %s"),
                    num_tries, url)
                raise
            else:
                # Pause for a bit and retry
                wait = initial_wait * 2**(num_tries - 1)
                rand_wait = random.randint(*rand_wait_range)
                total_wait = wait + rand_wait
                logger.info(("Encountered HTTPError or HTTPException or "
                    "URLError or timeout when opening url; sleeping for %d "
                    "seconds, and then trying again"), total_wait)
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
    params = [('taxid', taxid), ('cmd', 'download2')]
    if ncbi_api_key is not None:
        params += [('api_key', ncbi_api_key)]
    params_url = urllib.parse.urlencode(params)
    url = 'https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?%s' % params_url
    return url


def ncbi_reference_url(taxid):
    """Construct URL for downloading list of genome references.

    Args:
        taxid: taxonomic ID to download references for

    Returns:
        str representing download URL
    """
    params = [('taxid', taxid), ('cmd', 'download1')]
    if ncbi_api_key is not None:
        params += [('api_key', ncbi_api_key)]
    params_url = urllib.parse.urlencode(params)
    url = 'https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?%s' % params_url
    return url


def read_taxid_from_ncbi_url(url):
    """Read the taxid from a URL.

    Args:
        url: str representing URL for downloading NCBI neighbors

    Returns:
        taxid in url, or None if no taxid is found
    """
    url_query = urllib.parse.urlparse(url).query
    queries = urllib.parse.parse_qs(url_query)
    if 'taxid' not in queries:
        return None
    return int(queries['taxid'][0])


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

    # For some taxids, NCBI redirects to a different taxid and we are
    # then unable to read the contents of the response (because it is
    # HTML, and not the expected text format)
    # Check if this occurred by seeing if the taxid in the url of
    # the response is different from what was requested; if it is,
    # then re-fetch the url with the new taxid (only try this once, otherwise
    # we could end up in a loop of redirects)
    taxid_in_url = read_taxid_from_ncbi_url(r.geturl())
    if taxid_in_url != taxid:
        logger.warning(("The neighbors table for taxid %d is being redirected "
            "to taxid %d"), taxid, taxid_in_url)
        new_taxid = taxid_in_url
        url = ncbi_neighbors_url(new_taxid)
        r = urlopen_with_tries(url)

    raw_data = r.read()
    for line in raw_data.decode('utf-8').split('\n'):
        line_rstrip = line.rstrip()
        if line_rstrip != '':
            yield line_rstrip


def fetch_references(taxid):
    """Fetch genome references list from NCBI.

    Args:
        taxid: taxonomic ID to download references for

    Yields:
        lines, where each line is from the genome references
        list and each line is a str
    """
    logger.debug(("Fetching list of references for tax %d") % taxid)

    url = ncbi_reference_url(taxid)
    r = urlopen_with_tries(url)

    # For some taxids, NCBI redirects to a different taxid and we are
    # then unable to read the contents of the response (because it is
    # HTML, and not the expected text format)
    # Check if this occurred by seeing if the taxid in the url of
    # the response is different from what was requested; if it is,
    # then re-fetch the url with the new taxid (only try this once, otherwise
    # we could end up in a loop of redirects)
    taxid_in_url = read_taxid_from_ncbi_url(r.geturl())
    if taxid_in_url != taxid:
        logger.warning(("The references list for taxid %d is being redirected "
            "to taxid %d"), taxid, taxid_in_url)
        new_taxid = taxid_in_url
        url = ncbi_reference_url(new_taxid)
        r = urlopen_with_tries(url)

    raw_data = r.read()
    for line in raw_data.decode('utf-8').split('\n'):
        line_rstrip = line.rstrip()
        if line_rstrip != '':
            yield line_rstrip


def ncbi_influenza_genomes_url(database='genomeset'):
    """Construct URL for downloading NCBI influenza genomes database.

    Args:
        database: db to use; 'genomeset' or 'influenza_na'

    Returns:
        str representing download URL
    """
    url = 'ftp://ftp.ncbi.nih.gov/genomes/INFLUENZA/'
    assert database in ['genomeset', 'influenza_na']
    url += database + '.dat.gz'
    return url


def fetch_influenza_genomes_table(species_name, database):
    """Fetch influenza genome table from NCBI.

    Args:
        species_name: filter to keep only lines that contain this species
            name
        database: db to use; 'genomeset' or 'influenza_na'

    Yields:
        lines, where each line is from the genome database table and
        each line is a str
    """
    logger.debug(("Fetching table of influenza genomes for species %s") %
        species_name)
    species_name_lower = species_name.lower()

    url = ncbi_influenza_genomes_url(database)
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
    params = [('id', ids), ('db', 'nuccore'), ('rettype', 'fasta'),
                ('retmode', 'text')]
    if ncbi_api_key is not None:
        params += [('api_key', ncbi_api_key)]
    # Use safe=',' to not encode ',' as '%2'
    params_url = urllib.parse.urlencode(params, safe=',')
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?%s' % params_url
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

    if ncbi_api_key is not None:
        # Using an API keys allows more requests per second (up to 10)
        reqs_per_sec = 7

    # Make temp file
    fp = tempfile.NamedTemporaryFile(delete=False)

    # Download sequences in batches
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:(i + batch_size)]
        url = ncbi_fasta_download_url(batch)
        raw_data = urlopen_with_tries(url, read=True)
        for line in raw_data.decode('utf-8').split('\n'):
            fp.write((line + '\n').encode())
        time.sleep(1.0/reqs_per_sec)

    # Closes the file so that it can be reopened on Windows
    fp.close()

    return fp


def ncbi_xml_download_url(accessions):
    """Construct URL for downloading GenBank XML.

    Args:
        accessions: collection of accessions to download XML for

    Returns:
        str representing download URL
    """
    ids = ','.join(accessions)
    params = [('id', ids), ('db', 'nuccore'), ('retmode', 'xml')]
    if ncbi_api_key is not None:
        params += [('api_key', ncbi_api_key)]
    # Use safe=',' to not encode ',' as '%2'
    params_url = urllib.parse.urlencode(params, safe=',')
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?%s' % params_url
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
    fp = tempfile.NamedTemporaryFile(delete=False)

    # Only write the header once; otherwise, it will be written for each
    # beach, and then the file will not be able to be parsed
    def is_xml_header(line):
        return (line.startswith('<?xml ') or line.startswith('<!DOCTYPE '))

    # Download in batches
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:(i + batch_size)]
        url = ncbi_xml_download_url(batch)
        raw_data = urlopen_with_tries(url, read=True)
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

    # Closes the file so that it can be reopened on Windows
    fp.close()

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

def construct_references(taxid):
    """Construct reference accession number list for a taxonomic ID.

    Args:
        taxid: taxonomic ID to download references for

    Returns:
        list of strings of reference accession numbers
    """
    logger.info(("Constructing a list of references for tax %d") % taxid)

    references = []
    for line in fetch_references(taxid):
        ls = line.strip()
        if len(ls) == 0:
            continue
        references.append(ls)

    if len(references) == 0:
        raise ValueError("Taxonomic ID %d does not have any references in "
                         "NCBI. Try specifying a reference accession using "
                         "--ref-accs" %taxid)
    return references


def add_metadata_to_neighbors_and_filter(neighbors, meta_filt=None, meta_filt_against=None):
    """Fetch and add metadata to neighbors, and filter them based on metadata

    This only fetches for neighbors that do not have metadata set.

    This also filters out neighbors which do not match the filters in meta_filt or 
    match a filter in meta_filt_against.

    Args:
        neighbors: collection of Neighbor objects
        meta_filt: tuple of 2 dictionaries where the keys are any of 'country', 'year',
            'entry_create_year', 'taxid' and values for the first are a collection 
            of what to include or True to indicate that the metadata must exist and
            the second are what to exclude.
        meta_filt_against: tuple of 2 dictionaries where the keys are any of 'country', 
            'year', 'entry_create_year', 'taxid' and values for the first are a 
            collection of what to include in accessions to be specific against and
            the second are what to exclude.

    Returns:
        neighbors with metadata included (excluding the ones filtered out), and
            accession numbers of neighbors for the design to be specific against 
            (as specified by meta_filt_against)
    """
    # Fetch metadata for each neighbor without metadata
    to_fetch = set(n.acc for n in neighbors if n.metadata == {})
    if len(to_fetch) > 0:
        metadata = fetch_metadata(to_fetch)
    else:
        metadata = {}

    # Keep track of the number of neighbors filtered out by meta_filt_against and not 
    # meta_filt, as these may or may not be unintentional
    only_in_against = 0
    # Keep track of which neighbors don't match meta_filt or match meta_filt_against,
    # as these should not be included in the design
    acc_to_exclude = set()
    # Keep track of which neighbors match meta_filt_against, as the design should
    # be specific against these accessions
    specific_against_metadata_acc = set()

    # The first value of the filter is what the metadata should equal; the second is
    # what the metadata should not equal
    if meta_filt:
        meta_filt_eq, meta_filt_neq = meta_filt
    if meta_filt_against:
        meta_filt_against_eq, meta_filt_against_neq = meta_filt_against
    for neighbor in neighbors:
        # Set metadata for neighbor if it was not previously set
        if neighbor.acc in to_fetch:
            neighbor.metadata = metadata[neighbor.acc]

        # If there is a filter for metadata, exclude any neighbors that don't match it 
        # in the design
        if meta_filt:
            for key, value in meta_filt_eq.items():
                if value is True and neighbor.metadata[key] is None:
                    # Filter says that the metadata must exist, and it doesn't for this 
                    # neighbor, so exclude
                    acc_to_exclude.add(neighbor.acc)
                elif neighbor.metadata[key] not in value:
                    # Metadata doesn't match the list of what it should equal, so exclude
                    acc_to_exclude.add(neighbor.acc)
            for key, value in meta_filt_neq.items():
                if neighbor.metadata[key] in value:
                    # Metadata matches the list of what it shouldn't equal, so exclude
                    acc_to_exclude.add(neighbor.acc)

        # If there is a filter to be specific against metadata, add any neighbors that
        # match it to specific_against_metadata_acc and exclude them from the design
        if meta_filt_against:
            for key, value in meta_filt_against_eq.items():
                if neighbor.metadata[key] in value:
                    # Metadata matches the list of what should be designed against,
                    # so exclude if not already excluded & add to list to be specific against
                    if neighbor.acc not in acc_to_exclude:
                        acc_to_exclude.add(neighbor.acc)
                        only_in_against += 1
                    specific_against_metadata_acc.add(neighbor.acc)
            for key, value in meta_filt_against_neq.items():
                if neighbor.metadata[key] not in value:
                    # Metadata doesn't match the list of what shouldn't be designed against,
                    # so exclude if not already excluded & add to list to be specific against
                    if neighbor.acc not in acc_to_exclude:
                        acc_to_exclude.add(neighbor.acc)
                        only_in_against += 1
                    acc_to_exclude.add(neighbor.acc)
                    specific_against_metadata_acc.add(neighbor.acc)

    # Remove accessions that do not match the filters
    if len(acc_to_exclude) > 0:
        logger.info(("Leaving out %d accessions that do not match "
            "metadata filters, %d of which were only identified by "
            "specific_against_metadata_filter"), 
            len(acc_to_exclude), only_in_against)
    included_neighbors = [n for n in neighbors if n.acc not in acc_to_exclude]

    return included_neighbors, specific_against_metadata_acc


def construct_influenza_genome_neighbors(taxid):
    """Construct Neighbor objects for all influenza genomes

    According to the README on NCBI's influenza database FTP site:
    ```
    The influenza_na.dat and influenza_aa.dat files have an additional field in
    the last column to indicate the completeness of a sequence - "c" for
    complete sequences that include start and stop codons; "nc" for nearly
    complete sequences that are missing only start and/or stop codons; "p" for
    partial sequences.
    ```
    ```
    The genomeset.dat file contains information for sequences of viruses with
    a complete set of segments in full-length (or nearly full-length).  Those
    of the same virus are grouped together (using an internal group ID that is
    shown in the last column of the file) and separated by an empty line from
    those of other viruses.
    ```
    fetch_influenza_genomes_table() returns influenza_na.dat or
    genomeset.dat (as specified), filtered for a given species name.

    The columns are:
    ```
    GenBank accession number[tab]Host[tab]Genome segment number or protein name
    [tab]Subtype[tab]Country[tab]Year/month/date[tab]Sequence length
    [tab]Virus name[tab]Age[tab]Gender[tab]Completeness indicator (last field)
    ```
    (the last column is only in influenza_na.dat)

    This keeps only complete or near-complete sequences.

    Args:
        taxid: taxonomic ID for an influenza species; must be influenza A
            or B or C species

    Returns:
        list of Neighbor objects
    """
    logger.info(("Constructing a list of neighbors for influenza species "
                 "with tax %d") % taxid)

    influenza_species = {11320: 'Influenza A virus',
                         11520: 'Influenza B virus',
                         11552: 'Influenza C virus'}
    if taxid not in influenza_species:
        raise ValueError(("Taxid (%d) must be for either influenza A or "
                          "B or C virus species") % taxid)
    species_name = influenza_species[taxid]

    influenza_lineages = {11320: ('Orthomyxoviridae', 'Alphainfluenzavirus',
                                  'Influenza A virus'),
                          11520: ('Orthomyxoviridae', 'Betainfluenzavirus',
                                  'Influenza B virus'),
                          11552: ('Orthomyxoviridae', 'Gammainfluenzavirus',
                                  'Influenza C virus')}
    lineage = influenza_lineages[taxid]

    # Construct a pattern to match years in a date (1000--2999)
    year_p = re.compile('([1-2][0-9]{3})')

    # Determine the current year
    curr_year = int(datetime.datetime.now().year)

    # Choose a database to pull from; note that 11552 is only in
    # influenza_na. 11320 and 11520 are in influenza_na and
    # genomeset. influenza_na has more sequences, whereas
    # genomeset is about ~1/2 the size but more highly curated for
    # genomes
    if taxid == 11320 or taxid == 11520:
        database = 'genomeset'
    elif taxid == 11552:
        database = 'influenza_na'

    neighbors = []
    for line in fetch_influenza_genomes_table(species_name, database):
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

        if database == 'influenza_na':
            # These include partial sequences; cut them out - i.e., only
            # keep this sequence if it is complete or near-complete
            completeness = ls[-1]
            if completeness not in ['c', 'nc']:
                continue

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
        Note that in some cases, the qualifier value can be None.
    """
    doc = minidom.parse(fn)
    def parse_xml_node_value(element, tag_name):
        els = element.getElementsByTagName(tag_name)
        if len(els) == 0:
            return None
        else:
            return els[0].firstChild.nodeValue

    source_features = defaultdict(list)

    seqs = doc.getElementsByTagName('GBSeq')
    for seq in seqs:
        accession = parse_xml_node_value(seq, 'GBSeq_primary-accession')

        create_date = parse_xml_node_value(seq, 'GBSeq_create-date')
        source_features[accession].append(('create_date', create_date))

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
        dict {accession: {'country': country, 'year': collection-year, 
            'entry_create_year': create-year, 'taxid': taxonomic id}}
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
        entry_create_year = None
        country = None
        taxid = None
        for (name, value) in feats:
            if name == 'collection_date' or name == 'create_date':
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
                if name == 'collection_date':
                    year = feat_year
                if name == 'create_date':
                    entry_create_year = feat_year
            if name == 'country':
                if country is not None and value != country:
                    raise Exception(("Inconsistent country for "
                        "accession %s") % accession)
                country = value
            if name == 'db_xref':
                # Find a string of digits that represents the taxonomic ID
                taxid = int(re.search(r"\d+", value)[0])
        metadata[accession] = {'country': country, 'year': year,
                'entry_create_year': entry_create_year, 'taxid': taxid}

    # Delete the tempfile
    unlink(xml_tf.name)

    return metadata

