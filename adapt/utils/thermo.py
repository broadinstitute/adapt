import numpy as np
import math
import logging
import itertools

from adapt.utils.oligo import FASTA_CODES, make_complement, is_complement, is_symmetric, gc_frac

try:
    import primer3
except ImportError:
    thermo_props = False
else:
    thermo_props = True

logger = logging.getLogger(__name__)

# SantaLucia & Hicks (2004), Annu. Rev. Biophys. Biomol. Struct 33: 415-440
# dH (kcal/mol), dS (kcal/mol K)

# DNA_DNA_INIT = (0.2, -0.0057)

# DNA_DNA_SYM = (0, -0.0014)

# DNA_DNA_TERM_AT = (2.2, 0.0069)

# DNA_DNA_SALT = (0, 0.000368)

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
# Constants derived from Primer3 and MELTING software package
# Match thermodynamic constants are from Santalucia 1998; mismatch constants
# are from Santalucia 2004
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

# Santalucia 2004
# DOI:10.1146/annurev.biophys.32.110601.141800
# DNA_DNA_INTERNAL = {
#     'A': {
#         'A': {
#             'A': (1.2, 0.0017),
#             'T': (-7.6, -0.0213),
#             'C': (2.3, 0.0046),
#             'G': (-0.6, -0.0023),
#         },
#         'T': {
#             'A': (-7.2, -0.0204),
#             'T': (-2.7, -0.0108),
#             'C': (-1.2, -0.0062),
#             'G': (-2.5, -0.0083),
#         },
#         'C': {
#             'A': (5.3, 0.0146),
#             'T': (0.7, 0.0002),
#             'C': (0.0, -0.0044),
#             'G': (-8.4, -0.0224),
#         },
#         'G': {
#             'A': (-0.7, -0.0023),
#             'T': (1.0, 0.0009),
#             'C': (-7.8, -0.0210),
#             'G': (-3.1, -0.0095),
#         },
#     },
#     'T': {
#         'A': {
#             'A': (4.7, 0.0129),
#             'T': (-7.2, -0.0204),
#             'C': (3.4, 0.0080),
#             'G': (0.7, 0.0007),
#         },
#         'T': {
#             'A': (-7.6, -0.0213),
#             'T': (0.2, -0.0015),
#             'C': (1.0, 0.0007),
#             'G': (-1.3, -0.0053),
#         },
#         'C': {
#             'A': (7.6, 0.0202),
#             'T': (1.2, 0.0007),
#             'C': (6.1, 0.0164),
#             'G': (-8.2, -0.0222),
#         },
#         'G': {
#             'A': (3.0, 0.0074),
#             'T': (-0.1, -0.0017),
#             'C': (-8.5, -0.0227),
#             'G': (1.6, 0.0036),
#         },
#     },
#     'C': {
#         'A': {
#             'A': (-0.9, -0.0042),
#             'T': (-8.5, -0.0227),
#             'C': (1.9, 0.0037),
#             'G': (-0.7, -0.0023),
#         },
#         'T': {
#             'A': (-7.8, -0.0210),
#             'T': (-5.0, -0.0158),
#             'C': (-1.5, -0.0061),
#             'G': (-2.8, -0.0080),
#         },
#         'C': {
#             'A': (0.6, -0.0006),
#             'T': (-0.8, -0.0045),
#             'C': (-1.5, -0.0072),
#             'G': (-8.0, -0.0190),
#         },
#         'G': {
#             'A': (-4.0, -0.0132),
#             'T': (-4.1, -0.0117),
#             'C': (-10.6, -0.0272),
#             'G': (-4.9, -0.0153),
#         },
#     },
#     'G': {
#         'A': {
#             'A': (-2.9, -0.0098),
#             'T': (-8.2, -0.0222),
#             'C': (5.2, 0.0142),
#             'G': (-0.6, -0.0010),
#         },
#         'T': {
#             'A': (-8.4, -0.0224),
#             'T': (-2.2, -0.0084),
#             'C': (5.2, 0.0135),
#             'G': (-4.4, -0.0123),
#         },
#         'C': {
#             'A': (-0.7, -0.0038),
#             'T': (2.3, 0.0054),
#             'C': (3.6, 0.0089),
#             'G': (-10.6, -0.0272),
#         },
#         'G': {
#             'A': (0.5, 0.0032),
#             'T': (3.3, 0.0104),
#             'C': (-8.0, -0.0190),
#             'G': (-6.0, -0.0158),
#         },
#     },
# }

# Double mismatch constants derived from MELTING software package
# DNA_DNA_INTERNAL_DOUBLE = {
#     "GG/TT": (5.8, 0.0163),
#     "GT/TG": (4.1, 0.0095),
#     "TG/GT": (-1.4, -0.0062)
# }

# DNA_DNA_TERMINAL accounts for the thermodynamic properities of pairs of bases
# that are at the ends of the binding region, as long as there is <= 1
# mismatch in that pair
# If the sequences are
#   ...5'-WX-3'
#   ...3'-YZ-5'
# If W matches with Y, the thermodynamic properties can be found at DNA_DNA_INTERNAL[W][X][Z].
# If X matches with Z, the thermodynamic properties can be found at DNA_DNA_INTERNAL[Z][Y][W].
# Constants derived from Primer3 and MELTING software package
# Santalucia 1998 uses the same constants for terminal bases as internal ones;
# Santalucia 2004 uses different ones.

# Santalucia 1998
DNA_DNA_TERMINAL = DNA_DNA_INTERNAL

# Santalucia 2004
# DNA_DNA_TERMINAL = {
#     'A': {
#         'A': {
#             'A': (-3.1, -0.0078),
#             'T': (-7.6, -0.0213),
#             'C': (-1.6, -0.0040),
#             'G': (-1.9, -0.0044),
#         },
#         'T': {
#             'A': (-7.2, -0.0204),
#             'T': (-2.4, -0.0065),
#             'C': (-2.3, -0.0063),
#             'G': (-3.5, -0.0094),
#         },
#         'C': {
#             'A': (-1.8, -0.0038),
#             'T': (-0.9, -0.0017),
#             'C': (-0.1, 0.0005),
#             'G': (-8.4, -0.0224),
#         },
#         'G': {
#             'A': (-2.5, -0.0059),
#             'T': (-3.2, -0.0087),
#             'C': (-7.8, -0.0210),
#             'G': (-1.1, -0.0021),
#         },
#     },
#     'T': {
#         'A': {
#             'A': (-2.5, -0.0063),
#             'T': (-7.2, -0.0204),
#             'C': (-2.3, -0.0059),
#             'G': (-2.0, -0.0047),
#         },
#         'T': {
#             'A': (-7.6, -0.0213),
#             'T': (-3.2, -0.0089),
#             'C': (-0.7, -0.0012),
#             'G': (-3.6, -0.0098),
#         },
#         'C': {
#             'A': (-2.7, -0.0070),
#             'T': (-2.5, -0.0063),
#             'C': (-0.7, -0.0013),
#             'G': (-8.2, -0.0222),
#         },
#         'G': {
#             'A': (-2.4, -0.0058),
#             'T': (-3.9, -0.0105),
#             'C': (-8.5, -0.0227),
#             'G': (-1.1, -0.0027),
#         },
#     },
#     'C': {
#         'A': {
#             'A': (-4.3, -0.0107),
#             'T': (-8.5, -0.0227),
#             'C': (-2.6, -0.0059),
#             'G': (-3.9, -0.0096),
#         },
#         'T': {
#             'A': (-7.8, -0.0210),
#             'T': (-6.1, -0.0169),
#             'C': (-3.9, -0.0106),
#             'G': (-6.6, -0.0187),
#         },
#         'C': {
#             'A': (-2.7, -0.0060),
#             'T': (-3.2, -0.0080),
#             'C': (-2.1, -0.0051),
#             'G': (-8.0, -0.0190),
#         },
#         'G': {
#             'A': (-6.0, -0.0155),
#             'T': (-3.8, -0.0090),
#             'C': (-10.6, -0.0272),
#             'G': (-3.8, -0.0095),
#         },
#     },
#     'G': {
#         'A': {
#             'A': (-8.0, -0.0225),
#             'T': (-8.2, -0.0222),
#             'C': (-5.0, -0.0138),
#             'G': (-4.3, -0.0111),
#         },
#         'T': {
#             'A': (-8.4, -0.0224),
#             'T': (-7.4, -0.0212),
#             'C': (-3.0, -0.0078),
#             'G': (-5.9, -0.0161),
#         },
#         'C': {
#             'A': (-3.2, -0.0071),
#             'T': (-4.9, -0.0135),
#             'C': (-3.9, -0.0106),
#             'G': (-9.8, -0.0244),
#         },
#         'G': {
#             'A': (-4.6, -0.0114),
#             'T': (-5.7, -0.0159),
#             'C': (-8.0, -0.0190),
#             'G': (-0.7, -0.0192),
#         },
#     },
# }

# kcal/mol K
R_CONSTANT = .001987

# Adding this converts from Celsius to Kelvin
CELSIUS_TO_KELVIN = 273.15


class Conditions:
    """Thermodynamic conditions of the oligo binding reactions
    """
    def __init__(self, sodium=5e-2, magnesium=2.5e-3, dNTP=1.6e-3,
            oligo_concentration=3e-7, target_concentration=0, t=310.15):
        """
        Args:
            sodium: molar concentration of sodium ions.
            magnesium: molar concentration of magnesium ions. Only needed if
                magnesium concentration is greater than dNTP concentration.
            dNTP: molar concentration of dNTPs. Only needed if
                magnesium concentration is greater than dNTP concentration.
            oligo_concentration: molar concentration of oligos in reaction.
            target_concentration: molar concentration of target in reaction.
                Only needed if not significantly smaller than oligo
                concentration.
            t: temperature of the reaction.
        """
        self.sodium = sodium
        self.magnesium = magnesium
        self.dNTP = dNTP
        self.oligo_concentration = oligo_concentration
        self.target_concentration = target_concentration
        self.t = t


def _delta_g_from_h_s(h, s, t=310.15):
    """Get free energy from enthalpy and entropy

    Args:
        h: delta H (enthalpy) in kcal/mol
        s: delta S (entropy) in kcal/mol K
        t: temperature in K
    Returns:
        delta G (free energy) in kcal/mol
    """
    return h - t * s


def _avg_delta_h_s(thermo_table, w, x, z):
    return np.mean([np.mean([np.mean([thermo_table[w_i][x_j][z_k]
                                   for w_i in FASTA_CODES[w]], axis=0)
                          for x_j in FASTA_CODES[x]], axis=0)
                 for z_k in FASTA_CODES[z]], axis=0)


def binds(oligo_seq, target_seq, ideal_tm, delta_tm, reverse_oligo=True,
        conditions=Conditions()):
    """Determine whether an oligo binds to a target sequence.

    This tolerates ambiguity and decides whether an oligo binds based on
    whether its melting temperature is within a range (delta_tm) from an ideal
    melting temperature(ideal_tm).

    If the target sequence contains a gap (and the oligo sequence does
    not, as it should not), this decides that the oligo does not bind.

    TODO: Could speed up by terminating early in certain conditions, similarly
    to how oligo.binds terminates early

    Args:
        oligo_seq: str of an oligo sequence
        target_seq: str of a target sequence, same length as oligo_seq
        ideal_tm: float of the ideal melting temperature of the oligo
        delta_tm: float of how much the melting temperature of the oligo can
            vary from the ideal_tm and still be considered as binding
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object

    Returns:
        True if the binding melting temperature is within delta_tm of the
        ideal_tm
    """
    if '-' in target_seq:
      assert '-' not in oligo_seq
      return False
    tm = calculate_melting_temp(oligo_seq, target_seq, reverse_oligo,
        conditions)
    return abs(ideal_tm - tm) <= delta_tm


def calculate_melting_temp(oligo, target, reverse_oligo=True,
        conditions=Conditions(), saltmethod='santalucia'):
    """Calculate the melting temperature of the oligo binding to the target

    Uses the equations:
    Tm = (delta H) / (delta S + R * ln(|oligo|-|target|/2))
    Based on SantaLucia 2004
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object
        saltmethod: str of either 'santalucia' (default) or 'owczarzy'.
            'santalucia' uses the salt correction described in Santalucia 2004
            (DOI:10.1146/annurev.biophys.32.110601.141800, eq. 5) /
            von Ahsen 2001 (DOI:10.1093/clinchem/47.11.1956, eq. 7)
            'owczarzy' uses the salt correction described in Owczarzy 2004/2008
            (DOI:10.1021/bi034621r, eq. 22; DOI:10.1021/bi702363u, eq. 17).
            Anything else uses no salt correction
            TODO: 'owczarzy' still a bit buggy, possibly remove

    Returns:
        Melting temperature in K
    """
    try:
        if saltmethod=='santalucia':
            h,s = calculate_delta_h_s(oligo, target,
                                      reverse_oligo=reverse_oligo,
                                      conditions=conditions)
        else:
            h,s = calculate_delta_h_s(oligo, target,
                                      reverse_oligo=reverse_oligo,
                                      conditions=Conditions(
                                        sodium=1, magnesium=0, dNTP=0))
    except DoubleMismatchesError:
        return 0

    tm = h/(s + R_CONSTANT * np.log(
        conditions.oligo_concentration-conditions.target_concentration/2))

    if saltmethod=='owczarzy':
        K_a = 3e4
        D = (K_a*conditions.dNTP - K_a*conditions.magnesium + 1)**2 + 4 * K_a * conditions.magnesium
        free_mg = (-(K_a * conditions.dNTP - K_a * conditions.magnesium + 1) + D**.5)/(2*K_a)
        fgc = gc_frac(oligo)
        lnna = math.log(conditions.sodium)
        r = free_mg**.5/conditions.sodium
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
                a *= 0.843 - 0.352*(conditions.sodium**.5)*lnna
                d *= 1.279 - 4.03e-3*lnna - 8.03e-3*lnna**2
                g *= 0.486 - 0.258*lnna + 5.25e-3*lnna**3
            tm = (1/tm
                  + ((a - .91*lnmg
                      + fgc*(6.26 + d*lnmg)
                      + ((1/(2*(len(oligo)-1))) *
                         (-48.2 + 52.5*lnmg + g*(lnmg**2))))
                     * (10**-5)))**-1

    return tm


def calculate_percent_bound(oligo, target, reverse_oligo=True,
        conditions=Conditions()):
    """Calculate the percent of oligo bound to the target

    Uses the equations:
    K = [AB]/([A][B])
    [A_tot] = [A] + [AB]
    [B_tot] = [B] + [AB]
    which simplfies to:
    0 = K[AB]^2 - (K[A_tot]+K[B_tot]+1)[AB] + K[A_tot][B_tot]
    This is then solved with the quadratic formula
    Based on SantaLucia 2007

    Args:
        oligo: oligo sequence's perfect match target
        target: target sequence
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object

    Returns:
        Percent of oligo bound to the target
    """
    K = calculate_equilibrium_constant(target, oligo,
        reverse_oligo=reverse_oligo, conditions=conditions)

    # Coefficients for quadratic formula
    a = K
    b = -K*(conditions.oligo_concentration + conditions.target_concentration + 1)
    c = K*conditions.oligo_concentration*conditions.target_concentration
    ab_conc = np.roots((a, b, c))
    percent_bound = ab_conc/conditions.oligo_concentration
    percent_bound = percent_bound[percent_bound>=0]
    return percent_bound[percent_bound<=1]


def calculate_equilibrium_constant(oligo, target, reverse_oligo=True,
        conditions=Conditions()):
    """Calculate equilibrium constant of the target and oligo annealing

    Uses the equation:
    delta G = -RT*ln(K)
    which simplifies to
    K=e^{-delta G/(RT)}
    Based on SantaLucia 2007

    Args:
        oligo: oligo sequence's perfect match target
        target: target sequence
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object

    Returns:
        Equilibrium constant
    """
    delta_g = calculate_delta_g(oligo, target, reverse_oligo=reverse_oligo,
        conditions=conditions)
    return math.e**(-delta_g/(R_CONSTANT*conditions.t))


def calculate_delta_g(oligo, target, reverse_oligo=True,
        conditions=Conditions()):
    """Calculate free energy of the target and oligo annealing

    Based on SantaLucia et al 2004, von Ahsen et al 2001, MELTING, & Primer3
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800
    von Ahsen DOI: 10.1093/clinchem/47.11.1956

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object
    TODO: handling degeneracy properly

    Returns:
        Free energy in kcal/mol
    """

    h,s = calculate_delta_h_s(oligo, target, reverse_oligo=reverse_oligo,
                              conditions=conditions)

    return _delta_g_from_h_s(h, s, t=conditions.t)


def calculate_delta_h_s(oligo, target, reverse_oligo=True,
        conditions=Conditions()):
    """Calculate enthalpy and entropy of the target and oligo annealing

    Based on SantaLucia et al 2004, von Ahsen et al 2001, MELTING, & Primer3
    SantaLucia DOI:10.1146/annurev.biophys.32.110601.141800 (eq. 5)
    von Ahsen DOI:10.1093/clinchem/47.11.1956 (eq. 7)

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)
        conditions: a Conditions object
    TODO: handling degeneracy properly

    Returns:
        tuple of (enthalpy in kcal/mol, entropy in kcal/(mol K))
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
    if oligo[0] in FASTA_CODES['W']:
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]
    else:
        delta_h += DNA_DNA_TERM_GC[0]
        delta_s += DNA_DNA_TERM_GC[1]

    if oligo[-1] in FASTA_CODES['W']:
        delta_h += DNA_DNA_TERM_AT[0]
        delta_s += DNA_DNA_TERM_AT[1]
    else:
        delta_h += DNA_DNA_TERM_GC[0]
        delta_s += DNA_DNA_TERM_GC[1]

    def delta_h_s_increment(w, x, y, z, prev_match=None,
        thermo_prev_match=DNA_DNA_INTERNAL):
        """Calculate the enthalpy and entropy for 2 neighboring base pairs

        Utility function for nearest neighbor calculation

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

        Return:
            tuple of (enthalpy in kcal/mol of neighboring base pairs,
                entropy in kcal/(mol K) of neighboring base pairs)
        """
        curr_match = is_complement(x, z) == 1
        if prev_match is None:
            prev_match = is_complement(w, y) == 1
        if prev_match:
            return (_avg_delta_h_s(thermo_prev_match, w, x, z), curr_match)
        elif curr_match:
            return (_avg_delta_h_s(DNA_DNA_INTERNAL, z, y, w), curr_match)
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
    pos_ions = conditions.sodium + 120 * (max(
        (conditions.magnesium*1000 - conditions.dNTP*1000), 0)**0.5/1000)
    salt_corr = phosphates * np.log(pos_ions)

    delta_h += DNA_DNA_SALT[0] * salt_corr
    delta_s += DNA_DNA_SALT[1] * salt_corr
    return delta_h, delta_s


def calculate_i_x(target, oligo, reverse_oligo=True):
    """
    i_x is a linear score based on the distance of the nearest mismatch
    to the end, where a mismatch at the end is 6 and a mismatch 5 bp
    away from the end is 1. No mismatches in the 6 bps from the 3' end
    gives an i_x of 0.
    This is needed to use the Terminal Mismatch Model (TMM) for extension bias,
    as described in Döring 2019 (DOI:10.1038/s41598-019-47173-w)

    Args:
        target: target sequence
        oligo: oligo sequence's perfect match target
        reverse_oligo: True (default) if the oligo needs to be reverse
            complemented (if the oligo is a guide or a primer binding to
            the 3' end), False if the target needs to be reverse
            complemented (if the oligo is a primer binding to the 5' end)

    Returns:
        i_x of the oligo, as defined above
    """
    i_x = 0
    ol = len(oligo)
    for i in range(min(6, ol)):
        if reverse_oligo:
            # The 3' end of the reverse primer is at the start of the string, so
            # check from start of string
            bp = i
        else:
            # The 3' end of the forward primer is at the end of the string, so
            # check from end of string
            bp = ol - 1 - i
        if oligo[bp] != target[bp]:
            i_x = 6-i
            break

    return i_x


def has_no_secondary_structure(oligo, conditions):
    hairpin_dg = primer3.calcHairpin(oligo,
        mv_conc=conditions.sodium*1000,
        dv_conc=conditions.magnesium*1000,
        dntp_conc=conditions.dNTP*1000,
        dna_conc=conditions.oligo_concentration*10**9).dg/1000
    if hairpin_dg <= -3:
        return False
    # Homodimer
    homodimer_dg = primer3.calcHomodimer(oligo,
        mv_conc=conditions.sodium*1000,
        dv_conc=conditions.magnesium*1000,
        dntp_conc=conditions.dNTP*1000,
        dna_conc=conditions.oligo_concentration*10**9).dg/1000
    if homodimer_dg <= -6:
        return False
    return True


def has_no_heterodimers(oligo_set, conditions):
    for olg_i, olg_j in itertools.combinations(oligo_set, 2):
        heterodimer_dg = primer3.calcHeterodimer(olg_i, olg_j,
            mv_conc=conditions.sodium*1000,
            dv_conc=conditions.magnesium*1000,
            dntp_conc=conditions.dNTP*1000,
            dna_conc=conditions.oligo_concentration*10**9).dg/1000
        if heterodimer_dg <= -6:
            return False
    return True


class DoubleMismatchesError(ValueError):
    pass
