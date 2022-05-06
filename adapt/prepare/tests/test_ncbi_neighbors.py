"""Tests for ncbi_neighbors module.
"""

import unittest

from adapt.prepare import ncbi_neighbors as nn
from xml.dom import minidom
from collections import defaultdict

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestURLConstruction(unittest.TestCase):
    """Tests constructing URLs.
    """

    def test_ncbi_neighbors_url(self):
        url = nn.ncbi_neighbors_url(123)
        expected_url = ('https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup'
            '.cgi?taxid=123&cmd=download2')
        self.assertEqual(url, expected_url)

    def test_ncbi_reference_url(self):
        url = nn.ncbi_reference_url(123)
        expected_url = ('https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup'
            '.cgi?taxid=123&cmd=download1')
        self.assertEqual(url, expected_url)

    def test_ncbi_fasta_download_url(self):
        url = nn.ncbi_fasta_download_url(['A123', 'A456', 'B789'])
        expected_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch'
            '.fcgi?id=A123,A456,B789&db=nuccore&rettype=fasta&retmode=text')
        self.assertEqual(url, expected_url)

    def test_ncbi_xml_download_url(self):
        url = nn.ncbi_xml_download_url(['A123', 'A456', 'B789'])
        expected_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch'
            '.fcgi?id=A123,A456,B789&db=nuccore&retmode=xml')
        self.assertEqual(url, expected_url)

    def test_read_taxid(self):
        url = ('https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup'
            '.cgi?taxid=123&cmd=download2')
        taxid = nn.read_taxid_from_ncbi_url(url)
        expected_taxid = 123
        self.assertEqual(taxid, expected_taxid)

    def test_ncbi_taxonomy_url(self):
        url = nn.ncbi_taxonomy_url(64320)
        expected_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
            'esummary.fcgi?db=taxonomy&id=64320')
        self.assertEqual(url, expected_url)

    def test_ncbi_detail_taxonomy_url(self):
        url = nn.ncbi_detail_taxonomy_url(64320)
        expected_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
            'efetch.fcgi?db=taxonomy&id=64320')
        self.assertEqual(url, expected_url)

    def test_ncbi_search_taxonomy_url(self):
        url = nn.ncbi_search_taxonomy_url('Zika virus')
        expected_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
            'esearch.fcgi?db=taxonomy&rettype=xml&term=Zika+virus')
        self.assertEqual(url, expected_url)


class TestConstructNeighbors(unittest.TestCase):
    """Tests the construct_neighbors() function.

    The function construct_neighbors() calls fetch_neighbors_table(),
    which makes a request to NCBI. To avoid the request, this overrides
    fetch_neighbors_table() to return a known neighbors table.
    """

    def setUp(self):
        self.expected_table_contents = \
            ("## Comment line 1\n"
             "## Comment line 2\n"
             "## Columns:\tRepresentative\tNeighbor\tHost\tSelected lineage\tTaxonomy name\tSegment name\n"
             "NC_0123\tKY456\tvertebrate,human\tFamilyA,GenusA,SpeciesA\tSpeciesA\tsegment \n"
             "NC_0456\tAB123\tinvertebrate\tFamilyB,GenusB,SpeciesB\tSpeciesB\tsegment 1\n"
             "NC_0456\tAB456\tinvertebrate\tFamilyB,GenusB,SpeciesB\tSpeciesB\tsegment 2\n")
        self.expected_neighbors = [
            nn.Neighbor('KY456', 'NC_0123', ['vertebrate', 'human'],
                ('FamilyA', 'GenusA', 'SpeciesA'), 'SpeciesA', ''),
            nn.Neighbor('AB123', 'NC_0456', ['invertebrate'],
                ('FamilyB', 'GenusB', 'SpeciesB'), 'SpeciesB', '1'),
            nn.Neighbor('AB456', 'NC_0456', ['invertebrate'],
                ('FamilyB', 'GenusB', 'SpeciesB'), 'SpeciesB', '2')
        ]

        # Override nn.fetch_neighbors_table() to return expected_table_lines,
        # but keep the real function
        self.fetch_neighbors_table_real = nn.fetch_neighbors_table
        nn.fetch_neighbors_table = lambda taxid: self.expected_table_contents.split('\n')

    def test_construct_neighbors(self):
        neighbors = nn.construct_neighbors(123)
        self.assertEqual(neighbors, self.expected_neighbors)

    def tearDown(self):
        # Reset nn.fetch_neighbors_table()
        nn.fetch_neighbors_table = self.fetch_neighbors_table_real


class TestTaxonomies(unittest.TestCase):
    """Tests getting taxonomies/rank/subtaxa groups for accessions/taxons
    """
    def setUp(self):
        self.fetch_xml = nn.fetch_xml
        self.ncbi_search_taxonomy_url = nn.ncbi_search_taxonomy_url
        self.ncbi_detail_taxonomy_url = nn.ncbi_detail_taxonomy_url
        self.urlopen_with_tries = nn.urlopen_with_tries
        self.minidom_parse = minidom.parse
        minidom.parse = minidom.parseString

        def fake_fetch_xml(accessions):
            # Need to create a fake XML object with a 'name' with the XML file
            class Object(object):
                pass
            obj = Object()

            if len(accessions) == 1:
                # Just return Zika
                obj.name = ('<?xml version="1.0" encoding="UTF-8"  ?>\n'
                    '<!DOCTYPE GBSet PUBLIC "-//NCBI//NCBI GBSeq/EN" '
                    '"https://www.ncbi.nlm.nih.gov/dtd/NCBI_GBSeq.dtd">\n'
                    '<GBSet>\n'
                    '  <GBSeq>\n'
                    '    <GBSeq_accession-version>OK054351.1</GBSeq_accession-version>\n'
                    '    <GBSeq_organism>Zika virus</GBSeq_organism>\n'
                    '    <GBSeq_taxonomy>Viruses; Riboviria; Orthornavirae; '
                    'Kitrinoviricota; Flasuviricetes; Amarillovirales; '
                    'Flaviviridae; Flavivirus</GBSeq_taxonomy>\n'
                    '    <GBSeq_feature-table>\n'
                    '      <GBFeature>\n'
                    '        <GBFeature_key>source</GBFeature_key>\n'
                    '        <GBFeature_quals>\n'
                    '          <GBQualifier>\n'
                    '            <GBQualifier_name>db_xref</GBQualifier_name>\n'
                    '            <GBQualifier_value>taxon:64320</GBQualifier_value>\n'
                    '          </GBQualifier>\n'
                    '        </GBFeature_quals>\n'
                    '      </GBFeature>\n'
                    '    </GBSeq_feature-table>\n'
                    '  </GBSeq>\n'
                    '</GBSet>\n')
            else:
                # Return Zika and Dengue
                obj.name = ('<?xml version="1.0" encoding="UTF-8"  ?>\n'
                    '<!DOCTYPE GBSet PUBLIC "-//NCBI//NCBI GBSeq/EN" '
                    '"https://www.ncbi.nlm.nih.gov/dtd/NCBI_GBSeq.dtd">\n'
                    '<GBSet>\n'
                    '  <GBSeq>\n'
                    '    <GBSeq_accession-version>OK054351.1</GBSeq_accession-version>\n'
                    '    <GBSeq_organism>Zika virus</GBSeq_organism>\n'
                    '    <GBSeq_taxonomy>Viruses; Riboviria; Orthornavirae; '
                    'Kitrinoviricota; Flasuviricetes; Amarillovirales; '
                    'Flaviviridae; Flavivirus</GBSeq_taxonomy>\n'
                    '    <GBSeq_feature-table>\n'
                    '      <GBFeature>\n'
                    '        <GBFeature_key>source</GBFeature_key>\n'
                    '        <GBFeature_quals>\n'
                    '          <GBQualifier>\n'
                    '            <GBQualifier_name>db_xref</GBQualifier_name>\n'
                    '            <GBQualifier_value>taxon:64320</GBQualifier_value>\n'
                    '          </GBQualifier>\n'
                    '        </GBFeature_quals>\n'
                    '      </GBFeature>\n'
                    '    </GBSeq_feature-table>\n'
                    '  </GBSeq>\n'
                    '  <GBSeq>\n'
                    '    <GBSeq_accession-version>OK605599.1</GBSeq_accession-version>\n'
                    '    <GBSeq_organism>Dengue virus</GBSeq_organism>\n'
                    '    <GBSeq_taxonomy>Viruses; Riboviria; Orthornavirae; '
                    'Kitrinoviricota; Flasuviricetes; Amarillovirales; '
                    'Flaviviridae; Flavivirus</GBSeq_taxonomy>\n'
                    '    <GBSeq_feature-table>\n'
                    '      <GBFeature>\n'
                    '        <GBFeature_key>source</GBFeature_key>\n'
                    '        <GBFeature_quals>\n'
                    '          <GBQualifier>\n'
                    '            <GBQualifier_name>db_xref</GBQualifier_name>\n'
                    '            <GBQualifier_value>taxon:11070</GBQualifier_value>\n'
                    '          </GBQualifier>\n'
                    '        </GBFeature_quals>\n'
                    '      </GBFeature>\n'
                    '    </GBSeq_feature-table>\n'
                    '  </GBSeq>\n'
                    '</GBSet>\n')

            return obj

        nn.fetch_xml = fake_fetch_xml

        def fake_ncbi_search_taxonomy_url(taxon_name):
            if taxon_name == 'Zika virus':
                taxid = 64320
            else:
                taxid = 12637

            return ('<?xml version="1.0" encoding="UTF-8" ?>\n'
                '<!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD esearch '
                '20060628//EN" "https://eutils.ncbi.nlm.nih.gov/eutils/dtd/'
                '20060628/esearch.dtd">\n'
                '<eSearchResult><IdList>\n'
                '<Id>%i</Id>\n'
                '</IdList></eSearchResult>\n') %(taxid)

        nn.ncbi_search_taxonomy_url = fake_ncbi_search_taxonomy_url

        def fake_ncbi_detail_taxonomy_url(taxid):
            if taxid == 64320:
                virus = 'Zika'
            else:
                virus = 'Dengue'

            return ('<?xml version="1.0" ?>\n'
                '<!DOCTYPE TaxaSet PUBLIC "-//NLM//DTD Taxon, 14th January '
                '2002//EN" "https://www.ncbi.nlm.nih.gov/entrez/query/DTD/'
                'taxon.dtd">\n'
                '<TaxaSet><Taxon>\n'
                '    <ScientificName>%s virus</ScientificName>\n'
                '    <Rank>species</Rank>\n'
                '    <Lineage>Viruses; Riboviria; Orthornavirae; '
                'Kitrinoviricota; Flasuviricetes; Amarillovirales; '
                'Flaviviridae; Flavivirus</Lineage>\n'
                '    <LineageEx>\n'
                '        <Taxon>\n'
                '            <TaxId>11050</TaxId>\n'
                '            <ScientificName>Flaviviridae</ScientificName>\n'
                '            <Rank>family</Rank>\n'
                '        </Taxon>\n'
                '        <Taxon>\n'
                '            <TaxId>11051</TaxId>\n'
                '            <ScientificName>Flavivirus</ScientificName>\n'
                '            <Rank>genus</Rank>\n'
                '        </Taxon>\n'
                '    </LineageEx>\n'
                '</Taxon>\n\n'
                '</TaxaSet>\n') %virus

        nn.ncbi_detail_taxonomy_url = fake_ncbi_detail_taxonomy_url

        def fake_urlopen_with_tries(url, read=False):
            return url

        nn.urlopen_with_tries = fake_urlopen_with_tries

    def test_fetch_taxonomies(self):
        taxonomies = nn.fetch_taxonomies(['OK054351.1'])
        expected_taxonomies = defaultdict(list)
        expected_taxonomies['OK054351.1'] = (['Viruses', 'Riboviria',
            'Orthornavirae', 'Kitrinoviricota', 'Flasuviricetes',
            'Amarillovirales', 'Flaviviridae', 'Flavivirus', 'Zika virus'],
            64320)
        self.assertDictEqual(taxonomies, expected_taxonomies)

    def test_get_taxid(self):
        self.assertEqual(nn.get_taxid('Zika virus'),
                         64320)

    def test_get_taxonomy_name_of_rank(self):
        species = nn.get_taxonomy_name_of_rank(64320, 'species')
        self.assertEqual(species, 'Zika virus')
        subspecies = nn.get_taxonomy_name_of_rank(64320, 'subspecies')
        self.assertIsNone(subspecies)
        genus = nn.get_taxonomy_name_of_rank(64320, 'genus')
        self.assertEqual(genus, 'Flavivirus')

    def test_get_subtaxa_groups(self):
        # Zika & Dengue
        subtaxa_groups = nn.get_subtaxa_groups(['OK054351.1', 'OK605599.1'],
                                               'species')
        expected_subtaxa_groups = {'Dengue virus': ['OK605599.1'],
                                   'Zika virus': ['OK054351.1']}
        self.assertDictEqual(subtaxa_groups, expected_subtaxa_groups)

    def tearDown(self):
        # Reset functions
        nn.fetch_xml = self.fetch_xml
        nn.ncbi_search_taxonomy_url = self.ncbi_search_taxonomy_url
        nn.ncbi_detail_taxonomy_url = self.ncbi_detail_taxonomy_url
        nn.urlopen_with_tries = self.urlopen_with_tries
        minidom.parse = self.minidom_parse

