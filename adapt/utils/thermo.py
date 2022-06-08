#!/usr/bin/env python3

import numpy as np
import math
import logging
from adapt.utils.oligo import FASTA_CODES, make_complement, is_complement, is_symmetric, gc_frac

logger = logging.getLogger(__name__)

# Primer3 (SantaLucia 1998)
DNA_DNA_TERM_GC = (0.1, -0.0028)

DNA_DNA_SYM = (0, -0.0014)

DNA_DNA_TERM_AT = (2.3, 0.0041)

DNA_DNA_SALT = (0, 0.000368)

# DNA_DNA_INTERNAL accounts for the thermodynamic properities of pairs of bases
# that are not at the ends of the binding region, as long as there is <= 1
# mismatch in that pair
# If the sequences are
#   ...5'-WX-3'...
#   ...3'-YZ-5'...
# If W matches with Y, the thermodynamic properties can be found at DNA_DNA_INTERNAL[W][X][Z].
# If X matches with Z, the thermodynamic properties can be found at DNA_DNA_INTERNAL[Z][Y][W].
# Constants derived from Primer3 software package
DNA_DNA_INTERNAL = {
    'A': {
        'A': {
            'A': (1.2, 0.0017),
            'T': (-7.9, -0.0222),
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
            'T': (-7.2, -0.0213),
            'C': (3.4, 0.0080),
            'G': (0.7, 0.0007),
        },
        'T': {
            'A': (-7.9, -0.0222),
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
            'G': (-8.0, -0.0199),
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
            'G': (-9.8, -0.0244),
        },
        'G': {
            'A': (0.5, 0.0032),
            'T': (3.3, 0.0104),
            'C': (-8.0, -0.0199),
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
            'A': (1.2, 0.0017),
            'T': (-7.9, -0.0222),
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
            'T': (-7.2, -0.0213),
            'C': (3.4, 0.0080),
            'G': (0.7, 0.0007),
        },
        'T': {
            'A': (-7.9, -0.0222),
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
            'G': (-8.0, -0.0199),
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
            'G': (-9.8, -0.0244),
        },
        'G': {
            'A': (0.5, 0.0032),
            'T': (3.3, 0.0104),
            'C': (-8.0, -0.0199),
            'G': (-6.0, -0.0158),
        },
    },
}


# kcal/mol K
R_CONSTANT = .001987

def _delta_g_from_h_s(h, s, t=310.15):
    """
    Args:
        h: delta H (enthalpy) in kcal/mol
        s: delta S (entropy) in kcal/mol K
        t: temperature in K
    """
    return h - t * s


def binds(oligo_seq, target_seq, ideal_tm, delta_tm,
        reverse_oligo=False, sodium=5e-2, magnesium=0, dNTP=0,
        oligo_concentration=3e-7, target_concentration=0):
    if '-' in target_seq:
      assert '-' not in oligo_seq
      return False
    tm = calculate_melting_temp(oligo_seq, target_seq, reverse_oligo,
        sodium, magnesium, dNTP, oligo_concentration, target_concentration)
    return abs(ideal_tm - tm) <= delta_tm


def calculate_melting_temp(oligo, target, reverse_oligo=False, sodium=5e-2,
        magnesium=2.5e-3, dNTP=1.6e-3, oligo_concentration=3e-7,
        target_concentration=0, saltmethod='santalucia'):
    """
    Calculate the percent of oligo bound to the target

    Uses the equations:
    Tm = (delta H) / (delta S + R * ln(|oligo|-|target|/2))
    Based on SantaLucia 2004
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        magnesium: molar concentration of magnesium ions. Only needed if
            magnesium concentration is greater than dNTP concentration
        dNTP: molar concentration of dNTPs. Only needed if
            magnesium concentration is greater than dNTP concentration
        oligo_concentration: molar concentration of oligos in reaction.
        target_concentration: molar concentration of target in reaction.
            Only needed if not significantly smaller than oligo
            concentration.
    """
    try:
        if saltmethod=='santalucia':
            h,s = calculate_delta_h_s(target, oligo,
                                      reverse_oligo=reverse_oligo,
                                      sodium=sodium, magnesium=magnesium,
                                      dNTP=dNTP)
        else:
            h,s = calculate_delta_h_s(target, oligo,
                                      reverse_oligo=reverse_oligo,
                                      sodium=1, magnesium=0, dNTP=0)
    except DoubleMismatchesError:
        return 0

    tm = h/(s + R_CONSTANT * np.log(oligo_concentration-target_concentration/2))

    if saltmethod=='owczarzy':
        K_a = 3e4
        D = (K_a*dNTP - K_a*magnesium + 1)**2 + 4 * K_a * magnesium
        free_mg = (-(K_a * dNTP - K_a * magnesium + 1) + D**.5)/(2*K_a)
        fgc = gc_frac(oligo)
        lnna = math.log(sodium)
        r = free_mg**.5/sodium
        if r < .22:
            tm = (1/tm
                  + (((4.29*fgc-3.95)*lnna
                      + 0.940*(lnna**2))
                     * (10**-5)))**-1
        else:
            lnmg = math.log(free_mg)
            a = 3.92
            d = 1.42
            g = 8.31
            if r < 6:
                a *= 0.843 - 0.352*(sodium**.5)*lnna
                d *= 1.279 - 4.03e-3*lnna - 8.03e-3*lnna**2
                g *= 0.486 - 0.258*lnna + 5.25e-3*lnna**3
            tm = (1/tm
                  + ((a - .91*lnmg
                      + fgc*(6.26 + d*lnmg)
                      + ((1/(2*(len(oligo)-1))) *
                         (-48.2 + 52.5*lnmg + g*(lnmg**2))))
                     * (10**-5)))**-1

    return tm


def calculate_percent_bound(target, oligo, reverse_oligo=False, sodium=5e-2,
        magnesium=0, dNTP=0, oligo_concentration=3e-7,
        target_concentration=0, t=310.15):
    """
    Calculate the percent of oligo bound to the target

    Uses the equations:
    K = [AB]/([A][B])
    [A_tot] = [A] + [AB]
    [B_tot] = [B] + [AB]
    which simplfies to:
    0 = K[AB]^2 - (K[A_tot]+K[B_tot]+1)[AB] + K[A_tot][B_tot]
    This is then solved with the quadratic formula
    Based on SantaLucia 2007

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        magnesium: molar concentration of magnesium ions. Only needed if
            magnesium concentration is greater than dNTP concentration
        dNTP: molar concentration of dNTPs. Only needed if
            magnesium concentration is greater than dNTP concentration
        oligo_concentration: molar concentration of oligos in reaction.
        target_concentration: molar concentration of target in reaction.
            Only needed if not significantly smaller than oligo
            concentration.
        t: temperature in Kelvin
    """
    K = calculate_equilibrium_constant(target, oligo, reverse_oligo=False,
                                       sodium=sodium, magnesium=magnesium,
                                       dNTP=dNTP, t=310.15)

    # Coefficients for quadratic formula
    a = K
    b = -K*(oligo_concentration + target_concentration) + 1
    c = K*oligo_concentration*target_concentration

    return np.roots(a, b, c)


def calculate_equilibrium_constant(target, oligo, reverse_oligo=False,
        sodium=5e-2, magnesium=0, dNTP=0, t=310.15):
    """
    Calculate equilibrium constant of the target and oligo annealing

    Uses the equation:
    delta G = -RT*ln(K)
    which simplifies to
    K=e^{-delta G/(RT)}
    Based on SantaLucia 2007

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        magnesium: molar concentration of magnesium ions. Only needed if
            magnesium concentration is greater than dNTP concentration
        dNTP: molar concentration of dNTPs. Only needed if
            magnesium concentration is greater than dNTP concentration
        t: temperature in Kelvin
    """
    delta_g = calculate_delta_g(target, oligo, reverse_oligo=reverse_oligo,
                                sodium=sodium, magnesium=magnesium, dNTP=dNTP,
                                t=t)
    return math.e**(-delta_g/(R_CONSTANT*t))


def calculate_delta_g(target, oligo, reverse_oligo=False, sodium=5e-2,
        magnesium=0, dNTP=0, t=310.15):
    """
    Calculate free energy of the target and oligo annealing

    Based on SantaLucia et al 2004, von Ahsen et al 2001, MELTING, & Primer3
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800
    von Ahsen DOI: 10.1093/clinchem/47.11.1956

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        magnesium: molar concentration of magnesium ions. Only needed if
            magnesium concentration is greater than dNTP concentration
        dNTP: molar concentration of dNTPs. Only needed if
            magnesium concentration is greater than dNTP concentration
        t: temperature in Kelvin
    TODO: handling degeneracy properly
    """

    h,s = calculate_delta_h_s(target, oligo, reverse_oligo=reverse_oligo,
                              sodium=sodium, magnesium=magnesium, dNTP=dNTP)

    return _delta_g_from_h_s(h, s, t=t)


def calculate_delta_h_s(target, oligo, reverse_oligo=False, sodium=5e-2,
        magnesium=0, dNTP=0):
    """
    Calculate free energy of the target and oligo annealing

    Based on SantaLucia et al 2004, von Ahsen et al 2001, MELTING, & Primer3
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800
    von Ahsen DOI: 10.1093/clinchem/47.11.1956

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True if the oligo is binding to the 3' end of the
            amplicon, False (default) if the oligo is binding to the 5' end
        sodium: molar concentration of sodium ions
        magnesium: molar concentration of magnesium ions. Only needed if
            magnesium concentration is greater than dNTP concentration
        dNTP: molar concentration of dNTPs. Only needed if
            magnesium concentration is greater than dNTP concentration
    TODO: handling degeneracy properly
    """
    if reverse_oligo:
        forward_seq = target
        reverse_seq = make_complement(oligo)
    else:
        forward_seq = oligo
        reverse_seq = make_complement(target)

    # delta G of initiation
    delta_h = 0
    delta_s = 0

    # Terminal AT penalties
    if (forward_seq[0] in FASTA_CODES['W'] and
            reverse_seq[0] in FASTA_CODES['W']):
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]
    else:
        delta_h += DNA_DNA_TERM_GC[0]
        delta_s += DNA_DNA_TERM_GC[1]

    if (forward_seq[-1] in FASTA_CODES['W'] and
            reverse_seq[-1] in FASTA_CODES['W']):
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]
    else:
        delta_h += DNA_DNA_TERM_GC[0]
        delta_s += DNA_DNA_TERM_GC[1]

    def delta_h_s_increment(w, x, y, z, prev_match=None,
        thermo_prev_match=DNA_DNA_INTERNAL):
        """
        If the bases are:
            5'-wx-3'
            3'-yz-5'
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
            prev_match = is_complement(w, y) == 1
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

    for b in range(2, len(oligo)-1):
        increment, prev_match = delta_h_s_increment(*forward_seq[b-1:b+1],
            *reverse_seq[b-1:b+1], prev_match=prev_match)
        delta_h += increment[0]
        delta_s += increment[1]

    increment, prev_match = delta_h_s_increment(*forward_seq[-2:],
            *reverse_seq[-2:], prev_match=prev_match,
            thermo_prev_match=DNA_DNA_TERMINAL)
    delta_h += increment[0]
    delta_s += increment[1]

    # TODO: how to handle degeneracy. symmetric returns the percentage of
    # oligos that are perfectly symmetric, so this calculates the average delta
    # G amongst the possible oligos
    # Symmetry Correction
    symmetric = is_symmetric(oligo)
    delta_h += symmetric * DNA_DNA_SYM[0]
    delta_s += symmetric * DNA_DNA_SYM[1]

    # Salt Correction
    phosphates = len(forward_seq) - 1
    pos_ions = sodium + 120 * (max((magnesium*1000 - dNTP*1000), 0)**0.5/1000)
    salt_corr = phosphates * np.log(pos_ions)

    delta_h += DNA_DNA_SALT[0] * salt_corr
    delta_s += DNA_DNA_SALT[1] * salt_corr
    return delta_h, delta_s


class DoubleMismatchesError(ValueError):
    pass
