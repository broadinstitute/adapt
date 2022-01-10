import numpy as np
from adapt.utils.oligo import FASTA_CODES, make_complement, is_complement, is_symmetric

# Define entropic terms from Santalucia et al 2004
# DOI:10.1146/annurev.biophys.32.110601.141800

# SantaLucia & Hicks (2004), Annu. Rev. Biophys. Biomol. Struct 33: 415-440
# dH (kcal/mol), dS (kcal/mol K)

DNA_DNA_INIT = (0.2, -0.0057)

DNA_DNA_SYM = (0, -0.0014)

DNA_DNA_TERM_AT = (2.2, 0.0069)

DNA_DNA_SALT = (0, -0.000368)

# DNA_DNA_INTERNAL accounts for the thermodynamic properities of pairs of bases
# that are not at the ends of the binding region, as long as there is <= 1
# mismatch in that pair
# If the sequences are
#   ...5'-WX-3'...
#   ...3'-YZ-5'...
# If W matches with Y, the thermodynamic properties can be found at DNA_DNA_INTERNAL[W][X][Z].
# If X matches with Z, the thermodynamic properties can be found at DNA_DNA_INTERNAL[Z][Y][W].
# Constants derived from MELTING software package
DNA_DNA_INTERNAL = {
    'A': {
        'A': {
            'A': (1.2, 0.0017),
            'T': (-7.6, -0.0213),
            'C': (2.3, 0.0046),
            'G': (-0.6, -0.0023),
        },
        'T': {
            'A': (-7.2, -0.0204),
            'T': (-2.7, -0.0108),
            'C': (-1.2, -0.0062),
            'G': (-2.5, -0.0083),
        },
        'C': {
            'A': (5.3, 0.0146),
            'T': (0.7, 0.0002),
            'C': (0.0, -0.0044),
            'G': (-8.4, -0.0224),
        },
        'G': {
            'A': (-0.7, -0.0023),
            'T': (1.0, 0.0009),
            'C': (-7.8, -0.0210),
            'G': (-3.1, -0.0095),
        },
    },
    'T': {
        'A': {
            'A': (4.7, 0.0129),
            'T': (-7.2, -0.0204),
            'C': (3.4, 0.0080),
            'G': (0.7, 0.0007),
        },
        'T': {
            'A': (-7.6, -0.0213),
            'T': (0.2, -0.0015),
            'C': (1.0, 0.0007),
            'G': (-1.3, -0.0053),
        },
        'C': {
            'A': (7.6, 0.0202),
            'T': (1.2, 0.0007),
            'C': (6.1, 0.0164),
            'G': (-8.2, -0.0222),
        },
        'G': {
            'A': (3.0, 0.0074),
            'T': (-0.1, -0.0017),
            'C': (-8.5, -0.0227),
            'G': (1.6, 0.0036),
        },
    },
    'C': {
        'A': {
            'A': (-0.9, -0.0042),
            'T': (-8.5, -0.0227),
            'C': (1.9, 0.0037),
            'G': (-0.7, -0.0023),
        },
        'T': {
            'A': (-7.8, -0.0210),
            'T': (-5.0, -0.0158),
            'C': (-1.5, -0.0061),
            'G': (-2.8, -0.0080),
        },
        'C': {
            'A': (0.6, -0.0006),
            'T': (-0.8, -0.0045),
            'C': (-1.5, -0.0072),
            'G': (-8.0, -0.0190),
        },
        'G': {
            'A': (-4.0, -0.0132),
            'T': (-4.1, -0.0117),
            'C': (-10.6, -0.0272),
            'G': (-4.9, -0.0153),
        },
    },
    'G': {
        'A': {
            'A': (-2.9, -0.0098),
            'T': (-8.2, -0.0222),
            'C': (5.2, 0.0142),
            'G': (-0.6, -0.0010),
        },
        'T': {
            'A': (-8.4, -0.0224),
            'T': (-2.2, -0.0084),
            'C': (5.2, 0.0135),
            'G': (-4.4, -0.0123),
        },
        'C': {
            'A': (-0.7, -0.0038),
            'T': (2.3, 0.0054),
            'C': (3.6, 0.0089),
            'G': (-10.6, -0.0272),
        },
        'G': {
            'A': (0.5, 0.0032),
            'T': (3.3, 0.0104),
            'C': (-8.0, -0.0190),
            'G': (-6.0, -0.0158),
        },
    },
}

DNA_DNA_INTERNAL_DOUBLE = {
    "GG/TT": (5.8, 0.0163),
    "GT/TG": (4.1, 0.0095),
    "TG/GT": (-1.4, -0.0062)
}

DNA_DNA_TERMINAL = {
    'A': {
        'A': {
            'A': (-3.1, -0.0078),
            'T': (-7.6, -0.0213),
            'C': (-1.6, -0.0040),
            'G': (-1.9, -0.0044),
        },
        'T': {
            'A': (-7.2, -0.0204),
            'T': (-2.4, -0.0065),
            'C': (-2.3, -0.0063),
            'G': (-3.5, -0.0094),
        },
        'C': {
            'A': (-1.8, -0.0038),
            'T': (-0.9, -0.0017),
            'C': (-0.1, 0.0005),
            'G': (-8.4, -0.0224),
        },
        'G': {
            'A': (-2.5, -0.0059),
            'T': (-3.2, -0.0087),
            'C': (-7.8, -0.0210),
            'G': (-1.1, -0.0021),
        },
    },
    'T': {
        'A': {
            'A': (-2.5, -0.0063),
            'T': (-7.2, -0.0204),
            'C': (-2.3, -0.0059),
            'G': (-2.0, -0.0047),
        },
        'T': {
            'A': (-7.6, -0.0213),
            'T': (-3.2, -0.0089),
            'C': (-0.7, -0.0012),
            'G': (-3.6, -0.0098),
        },
        'C': {
            'A': (-2.7, -0.0070),
            'T': (-2.5, -0.0063),
            'C': (-0.7, -0.0013),
            'G': (-8.2, -0.0222),
        },
        'G': {
            'A': (-2.4, -0.0058),
            'T': (-3.9, -0.0105),
            'C': (-8.5, -0.0227),
            'G': (-1.1, -0.0027),
        },
    },
    'C': {
        'A': {
            'A': (-4.3, -0.0107),
            'T': (-8.5, -0.0227),
            'C': (-2.6, -0.0059),
            'G': (-3.9, -0.0096),
        },
        'T': {
            'A': (-7.8, -0.0210),
            'T': (-6.1, -0.0169),
            'C': (-3.9, -0.0106),
            'G': (-6.6, -0.0187),
        },
        'C': {
            'A': (-2.7, -0.0060),
            'T': (-3.2, -0.0080),
            'C': (-2.1, -0.0051),
            'G': (-8.0, -0.0190),
        },
        'G': {
            'A': (-6.0, -0.0155),
            'T': (-3.8, -0.0090),
            'C': (-10.6, -0.0272),
            'G': (-3.8, -0.0095),
        },
    },
    'G': {
        'A': {
            'A': (-8.0, -0.0225),
            'T': (-8.2, -0.0222),
            'C': (-5.0, -0.0138),
            'G': (-4.3, -0.0111),
        },
        'T': {
            'A': (-8.4, -0.0224),
            'T': (-7.4, -0.0212),
            'C': (-3.0, -0.0078),
            'G': (-5.9, -0.0161),
        },
        'C': {
            'A': (-3.2, -0.0071),
            'T': (-4.9, -0.0135),
            'C': (-3.9, -0.0106),
            'G': (-9.8, -0.0244),
        },
        'G': {
            'A': (-4.6, -0.0114),
            'T': (-5.7, -0.0159),
            'C': (-8.0, -0.0190),
            'G': (-0.7, -0.0192),
        },
    },
}


def _delta_g_from_h_s(h, s, t=310.15):
    """
    Args:
        h: delta H (enthalpy) in kcal/mol
        s: delta S (entropy) in kcal/mol K
        t: temperature in K
    """
    return h - t * s


def calculate_delta_g(target, oligo, reverse_oligo=False, sodium=1, t=310.15):
    """
    Calculate free energy of the target and oligo annealing

    Based on SantaLucia et al 2004 & MELTING
    DOI:10.1146/annurev.biophys.32.110601.141800

    Args:
        target: target sequence
        oligo: oligo sequence
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        t: temperature in Kelvin
    TODO: handling degeneracy properly
    """
    if reverse_oligo:
        forward_seq = target
        reverse_seq = make_complement(oligo)
    else:
        forward_seq = oligo
        reverse_seq = make_complement(target)

    # delta G of initiation
    delta_g = _delta_g_from_h_s(*DNA_DNA_INIT, t)

    def delta_g_increment(w, x, y, z, prev_match=None,
        thermo_prev_match=DNA_DNA_INTERNAL):
        """
        If the bases are:
            5'-wx-3'
            3'-yz'5'
        this will return the thermodynamic properties of this set of bases.

        Args:
            w: 5' base of 5' to 3' sequence
            x: 3' base of 5' to 3' sequence
            y: 3' base of 3' to 5' sequence
            z: 5' base of 3' to 5' sequence
            prev_match: Boolean if w and y match or not; None (default) checks
                in the function
            thermo_prev_match: If prev_match, use this dictionary of
                thermodyamic values. Otherwise, use the DNA_DNA_INTERNAL
                thermodynamic table. Defaults to DNA_DNA_INTERNAL
        """
        curr_match = is_complement(x, z) == 1
        if prev_match is None:
            curr_match = is_complement(w, y) == 1
        if prev_match:
            return (_delta_g_from_h_s(*thermo_prev_match[w][x][z], t),
                    curr_match)
        elif curr_match:
            return (_delta_g_from_h_s(*DNA_DNA_INTERNAL[z][y][w], t),
                    curr_match)
        else:
            raise DoubleMismatchesError("delta G cannot be calculated if there "
                "are two mismatches next to each other.")

    # delta G of starting base pair
    increment, prev_match = delta_g_increment(*reverse_seq[1::-1],
        *forward_seq[1::-1], thermo_prev_match=DNA_DNA_TERMINAL)
    delta_g += increment

    for b in range(1, len(oligo)-1):
        increment, prev_match = delta_g_increment(*forward_seq[b-1:b+1],
            *reverse_seq[b-1:b+1], prev_match=prev_match)
        delta_g += increment

    increment, prev_match = delta_g_increment(*forward_seq[-2:],
            *reverse_seq[-2:], prev_match=prev_match,
            thermo_prev_match=DNA_DNA_TERMINAL)
    delta_g += increment

    # Terminal AT penalties
    if (forward_seq[0] in FASTA_CODES['W'] and
            reverse_seq[0] in FASTA_CODES['W']):
        delta_g += _delta_g_from_h_s(*DNA_DNA_TERM_AT, t)
    if (forward_seq[-1] in FASTA_CODES['W'] and
            reverse_seq[-1] in FASTA_CODES['W']):
        delta_g += _delta_g_from_h_s(*DNA_DNA_TERM_AT, t)

    # TODO - how to handle degeneracy. is_symmetric returns the percentage of
    # oligos that are perfectly symmetric, so this calculates the average delta
    # G amongst the possible oligos
    # Symmetry Correction
    delta_g += is_symmetric(oligo) * _delta_g_from_h_s(*DNA_DNA_SYM, t)

    # Salt Correction
    phosphates = len(forward_seq) - 1
    delta_g = -(_delta_g_from_h_s(*DNA_DNA_SALT, t) * phosphates *
                np.log(sodium))

    return delta_g


def calculate_delta_h_s(target, oligo, reverse_oligo=False, sodium=1):
    """
    Calculate free energy of the target and oligo annealing

    Based on SantaLucia et al 2004 & MELTING
    DOI:10.1146/annurev.biophys.32.110601.141800

    Args:
        target: target sequence
        oligo: oligo sequence
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
    TODO: handling degeneracy properly
    """
    if reverse_oligo:
        forward_seq = target
        reverse_seq = make_complement(oligo)
    else:
        forward_seq = oligo
        reverse_seq = make_complement(target)

    # delta G of initiation
    delta_h, delta_s = *DNA_DNA_INIT

    def delta_h_s_increment(w, x, y, z, prev_match=None,
        thermo_prev_match=DNA_DNA_INTERNAL):
        """
        If the bases are:
            5'-wx-3'
            3'-yz'5'
        this will return the thermodynamic properties of this set of bases.

        Args:
            w: 5' base of 5' to 3' sequence
            x: 3' base of 5' to 3' sequence
            y: 3' base of 3' to 5' sequence
            z: 5' base of 3' to 5' sequence
            prev_match: Boolean if w and y match or not; None (default) checks
                in the function
            thermo_prev_match: If prev_match, use this dictionary of
                thermodyamic values. Otherwise, use the DNA_DNA_INTERNAL
                thermodynamic table. Defaults to DNA_DNA_INTERNAL
        """
        curr_match = is_complement(x, z) == 1
        if prev_match is None:
            curr_match = is_complement(w, y) == 1
        if prev_match:
            return (thermo_prev_match[w][x][z], curr_match)
        elif curr_match:
            return (DNA_DNA_INTERNAL[z][y][w], curr_match)
        else:
            raise DoubleMismatchesError("delta H/S cannot be calculated with "
                                        "two mismatches next to each other.")

    # delta G of starting base pair
    increment, prev_match = delta_h_s_increment(*reverse_seq[1::-1],
        *forward_seq[1::-1], thermo_prev_match=DNA_DNA_TERMINAL)
    delta_h += increment[0]
    delta_s += increment[1]

    for b in range(1, len(oligo)-1):
        increment, prev_match = delta_h_s_increment(*forward_seq[b-1:b+1],
            *reverse_seq[b-1:b+1], prev_match=prev_match)
        delta_h += increment[0]
        delta_s += increment[1]

    increment, prev_match = delta_h_s_increment(*forward_seq[-2:],
            *reverse_seq[-2:], prev_match=prev_match,
            thermo_prev_match=DNA_DNA_TERMINAL)
    delta_h += increment[0]
    delta_s += increment[1]

    # Terminal AT penalties
    if (forward_seq[0] in FASTA_CODES['W'] and
            reverse_seq[0] in FASTA_CODES['W']):
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]
    if (forward_seq[-1] in FASTA_CODES['W'] and
            reverse_seq[-1] in FASTA_CODES['W']):
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]

    # TODO - how to handle degeneracy. symmetric returns the percentage of
    # oligos that are perfectly symmetric, so this calculates the average delta
    # G amongst the possible oligos
    # Symmetry Correction
    symmetric = is_symmetric(oligo)
    delta_h += symmetric * DNA_DNA_SYM[0]
    delta_s += symmetric * DNA_DNA_SYM[1]

    # Salt Correction
    phosphates = len(forward_seq) - 1
    salt_corr = phosphates * np.log(sodium)
    # No delta H change for salt, only delta S
    delta_s = DNA_DNA_SALT[1] * phosphates * salt_corr

    return delta_h, delta_s


class DoubleMismatchesError(ValueError):
    pass
