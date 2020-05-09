"""Functions for clustering sequences before input.

This includes computing a distance matrix using MinHash, and
clustering that matrix.
"""

from collections import defaultdict
from collections import OrderedDict
import logging
import operator

import numpy as np
from scipy.cluster import hierarchy

from adapt.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def make_signatures_with_minhash(family, seqs):
    """Construct a signature using MinHash for each sequence.

    Args:
        family: lsh.MinHashFamily object
        seqs: dict mapping sequence header to sequences

    Returns:
        dict mapping sequence header to signature
    """
    # Construct a single hash function; use the same for all sequences
    h = family.make_h()

    signatures = {}
    for name, seq in seqs.items():
        signatures[name] = h(seq)
    return signatures


def create_condensed_dist_matrix(n, dist_fn):
    """Construct a 1d condensed distance matrix for scipy.

    Args:
        n: number of elements whose pairwise distances to store in the
            matrix
        dist_fn: function such that dist_fn(i, j) gives the distance
            between i and j, for all i<j<n

    Returns:
        condensed 1d distance matrix for input to scipy functions
    """
    def idx(i, j):
        # Compute index in 1d vector for pair (i, j)
        return int((-1 * i*i)/2 + i*n - 3*i/2 + j - 1)

    dist_matrix_len = int(n*(n-1)/2)
    dist_matrix = np.zeros(dist_matrix_len)

    for j in range(n):
        for i in range(j):
            dist_matrix[idx(i, j)] = dist_fn(i, j)

    return dist_matrix


def cluster_from_dist_matrix(dist_matrix, threshold, num_clusters=None):
    """Use scipy to cluster a distance matrix.

    Args:
        dist_matrix: distance matrix, represented in scipy's 1d condensed form
        threshold: maximum inter-cluster distance to merge clusters (higher
            results in fewer clusters)
        num_clusters: if set, cluster such that this is the maximum number
            of clusters; threshold is ignored and must be None

    Returns:
        list c such that c[i] is a collection of all the observations
        (whose pairwise distances are indexed in dist) in the i'th
        cluster, in sorted order by descending cluster size
    """
    if threshold is None and num_clusters is None:
        raise ValueError(("threshold or num_clusters must be set"))
    elif threshold is not None and num_clusters is not None:
        raise ValueError(("Only one of threshold or num_clusters can be "
            "set"))

    linkage = hierarchy.linkage(dist_matrix, method='average')
    if threshold:
        clusters = hierarchy.fcluster(linkage, threshold, criterion='distance')
    elif num_clusters:
        clusters = hierarchy.fcluster(linkage, num_clusters,
            criterion='maxclust')

    # clusters are numbered starting at 1, but base the count on
    # first_clust_num just in case this changes
    first_clust_num = min(clusters)
    num_clusters_found = max(clusters) + 1 - first_clust_num
    elements_in_cluster = defaultdict(list)
    for i, clust_num in enumerate(clusters):
        elements_in_cluster[clust_num].append(i)
    cluster_sizes = {c: len(elements_in_cluster[c])
                     for c in range(first_clust_num,
                                    num_clusters_found + first_clust_num)}

    elements_in_cluster_sorted = []
    for clust_num, _ in sorted(cluster_sizes.items(),
            key=operator.itemgetter(1), reverse=True):
        elements_in_cluster_sorted += [elements_in_cluster[clust_num]]
    return elements_in_cluster_sorted


def cluster_with_minhash_signatures(seqs, k=12, N=100, threshold=0.1,
        num_clusters=None, return_dist_matrix_and_indices=False):
    """Cluster sequences based on their MinHash signatures.

    Args:
        seqs: dict mapping sequence header to sequences
        k: k-mer size to use for k-mer hashes (smaller is likely more
            sensitive for divergent genomes, but may lead to false positives
            in determining which genomes are close)
        N: number of hash values to use in a signature (higher is slower for
            clustering, but likely more sensitive for divergent genomes)
        threshold: maximum inter-cluster distance to merge clusters, in
            average nucleotide dissimilarity (1-ANI, where ANI is
            average nucleotide identity); higher results in fewer
            clusters
        num_clusters: if set, cluster such that this is the maximum number
            of clusters; threshold is ignored and must be None
        return_dist_matrix_and_indices: if set, return the pairwise distance
            matrix and sequences indices in each cluster; when used,
            seqs should be an OrderedDict

    Returns:
        if return_dist_matrix_and_indices:
            tuple (dm, c) such that dm is the pairwise distance matrix and
            c is a list such that c[i] gives a collection of sequence
            indices (corresponding to indices in dm) in the same cluster
            (note dm is a 1d condensed matrix in scipy's form); clusters in
            c are sorted in descending order of size
        else:
            list c such that c[i] gives a collection of sequence headers
            in the same cluster, and the clusters in c are sorted
            in descending order of size
    """
    if len(seqs) == 1:
        # Simply return one cluster
        if return_dist_matrix_and_indices:
            return None, [[0]]
        else:
            return [[list(seqs.keys())[0]]]

    family = lsh.MinHashFamily(k, N=N)
    signatures_map = make_signatures_with_minhash(family, seqs)

    # Map each sequence header to an index (0-based), and get
    # the signature for the corresponding index
    num_seqs = len(seqs)
    seq_headers = []
    signatures = []
    for name, seq in seqs.items():
        seq_headers += [name]
        signatures += [signatures_map[name]]

    # Eq. 4 of the Mash paper (Ondov et al. 2016) shows that the
    # Mash distance, which is shown to be closely related to 1-ANI, is:
    #  D = (-1/k) * ln(2*j/(1+j))
    # where j is a Jaccard similarity. Solving for j:
    #  j = 1/(2*exp(k*D) - 1)
    # So, for a desired distance D in terms of 1-ANI, the corresponding
    # Jaccard distance is:
    #  1.0 - 1/(2*exp(k*D) - 1)
    # We can use this to calculate a clustering threshold in terms of
    # Jaccard distance
    if threshold is not None:
        jaccard_dist_threshold = 1.0 - 1.0/(2.0*np.exp(k*threshold) - 1)
    else:
        # Ignore inter-cluster distance; use num_clusters instead
        jaccard_dist_threshold = None

    def jaccard_dist(i, j):
        # Return estimated Jaccard dist between signatures at
        # index i and index j
        return family.estimate_jaccard_dist(
            signatures[i], signatures[j])

    dist_matrix = create_condensed_dist_matrix(num_seqs, jaccard_dist)
    clusters = cluster_from_dist_matrix(dist_matrix,
        jaccard_dist_threshold, num_clusters=num_clusters)

    if return_dist_matrix_and_indices:
        return dist_matrix, clusters
    else:
        seqs_in_cluster = []
        for cluster_idxs in clusters:
            seqs_in_cluster += [[seq_headers[i] for i in cluster_idxs]]
        return seqs_in_cluster


def find_representative_sequences(seqs, k=12, N=100, threshold=0.1,
        num_clusters=None, frac_to_cover=1.0):
    """Find a set of representative sequences.

    This clusters seqs, and then determines a medoid for each cluster.
    It returns the medoids.

    This will not return representative sequences with ambiguity or NNNs.

    Args:
        seqs, k, N, threshold, num_clusters: see cluster_with_minhash_signatures()
        frac_to_cover: return medoids from clusters that collectively
            account for at least this fraction of all sequences; this
            allows ignoring representative sequences for outlier
            clusters

    Returns:
        tuple (set of sequence headers representing cluster medoids,
        fraction of all sequences contained in cluster)
    """
    seqs = OrderedDict(seqs)
    dist_matrix, clusters = cluster_with_minhash_signatures(
            seqs, k=k, N=N, threshold=threshold, num_clusters=num_clusters,
            return_dist_matrix_and_indices=True)

    seqs_items = list(seqs.items())
    n = len(seqs)
    def idx(i, j):
        # Compute index in 1d vector for pair (i, j)
        if i > j:
            i, j = j, i
        return int((-1 * i*i)/2 + i*n - 3*i/2 + j - 1)

    rep_seqs = []
    rep_seqs_frac = []
    num_seqs_accounted_for = 0
    for cluster_idxs in clusters:
        # Stop if we have already accounted for frac_to_cover of the
        # sequences
        # Note that clusters should be sorted in descending order of
        # size, so any clusters after this one will be the same size
        # or smaller
        if float(num_seqs_accounted_for) / len(seqs) >= frac_to_cover:
            break

        # Find the medoid of this cluster
        # Simply look over all pairs in the cluster (there are faster
        # algorithms, though not linear)
        curr_medoid = None
        curr_medoid_dist_total = None
        for i in cluster_idxs:
            # Only allow i to be the medoid if it does not have ambiguity
            seq = seqs_items[i][1]
            if sum(seq.count(b) for b in ('A','C','G','T')) != len(seq):
                # Has ambiguity or NNNs; skip
                continue

            # Compute the total distance to all other sequences in this
            # cluster, and check if this is the medoid
            dist_total = 0
            for j in cluster_idxs:
                if i == j:
                    continue
                dist_total += dist_matrix[idx(i,j)]
            if curr_medoid is None or dist_total < curr_medoid_dist_total:
                curr_medoid = i
                curr_medoid_dist_total = dist_total
        if curr_medoid is not None:
            rep_seqs += [curr_medoid]
            rep_seqs_frac += [float(len(cluster_idxs)) / len(seqs)]
            num_seqs_accounted_for += len(cluster_idxs)
        else:
            # All sequences have ambiguity or NNNs; raise a warning and
            # skip this cluster
            logger.warning(("Cannot find medoid for cluster of size %d "
                "because all sequences have ambiguity or NNNs; skipping "
                "this cluster"),
                len(cluster_idxs))

    return ([seqs_items[i][0] for i in rep_seqs], rep_seqs_frac)
