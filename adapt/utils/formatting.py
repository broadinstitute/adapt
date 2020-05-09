"""Functions for formatting text.
"""

import logging

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def tag_seq_overlap(overlap_labels, target_seq):
    """Add tags around where a sequence overlaps a target.

    For example, if the target seq is 'ATCGAATTCCGG' and the overlap is the
    indices {1,2,5,6,7}, the output is
    'A<label>TC</label>GA<label>ATT</label>CCGG'.

    Args:
        overlap_labels: dict {label: set of indices to be tagged with 'label'}
        target_seq: str of a target sequence

    Returns:
        target_seq, with tags according to overlap_labels
    """
    target_seq_tagged = ''
    curr_label = None
    for i, base in enumerate(target_seq):
        # Check that position i is in at most 1 label
        ignore = set()
        found_overlap = False
        for label, overlaps in overlap_labels.items():
            if i in overlaps:
                if found_overlap:
                    # position i is in two different
                    # labels (curr_label and label)
                    # This can happen in some edge cases (e.g., indel in
                    # a target relative to the design); warn and choose one
                    # label
                    logger.warning(("Position in target sequence is "
                        "tagged with two labels; picking one"))
                    ignore.add(label)
                found_overlap = True

        # Start or close a tag
        for label, overlaps in overlap_labels.items():
            if label in ignore:
                continue
            if i in overlaps:
                # position i is in a tag
                if curr_label is None:
                    # Start the tag
                    target_seq_tagged += '<' + label + '>'
                    curr_label = label
                else:
                    if curr_label != label:
                        # position i is tagged with 'label', but not
                        # 'curr_label'; end curr_label and start label
                        target_seq_tagged += ('</' + curr_label + '>' +
                            '<' + label + '>')
                        curr_label = label
                    else:
                        # Simply within a tag; do nothing
                        pass
            else:
                if curr_label == label:
                    # position i is not in this tag anymore; close the tag
                    target_seq_tagged += '</' + curr_label + '>'
                    curr_label = None

        # Add the base
        target_seq_tagged += base

    # Close a tag, if one is open, at the end of the sequence
    if curr_label is not None:
        target_seq_tagged += '</' + curr_label + '>'
        curr_label = None

    return target_seq_tagged
