# Modules
import numpy as np
import astropy.units as u # manages units
from astropy.constants import sigma_sb, h, R_sun

def sigma_nu(nu):
    """Photoionisation cross section for hydrogen approximated as a power law nu^-3.

    Parameters
    ----------
    nu : float
        Frequency of incoming radiation

    Returns
    -------
    float
        The photoionisation cross section for hydrogen for frequency nu
    """
    nu_0 = (13.6 * u.eV)/h
    sigma_nu_0 = 6.33e-18 * u.cm**2 # from https://w.astro.berkeley.edu/~ay216/06/NOTES/ay216_2006_04_HII.pdf
    return sigma_nu_0 * (nu_0/nu)**3

def ionisation_fraction(gamma, n_H, alpha_B, debug = True):
    """Calculate ionisation fraction by solving recombination-ionisation balance.

    Parameters
    ----------
    gamma : float
        Ionisation rate per photon
    n_H : float
        Hydrogen number density
    alpha_B : float
        Recombination coefficient for case B "on the spot" recombination.
    debug : bool
        Debug flag to enable checks for physical result.

    Returns
    -------
    float
        The plus solution. Checks if is physical, in the interval [0, 1].
    """
    den = 2*n_H*alpha_B
    dis = gamma**2 + 2*den*gamma
    X = (-gamma + np.sqrt(dis)) / den

    if debug:
        if 0 < X and X < 1:
            return X
        else:
            raise(BaseException("Non physical solution! Value obtained: %f"%(X)))
    
    return X

def ionisation_fraction_B(gamma, n_H, alpha_B, debug = False):
    """Calculate ionisation fraction by solving recombination-ionisation balance. This version is cleaner but
    might lead to NaN values if gamma is very small.

    Parameters
    ----------
    gamma : float
        Ionisation rate per photon
    n_H : float
        Hydrogen number density
    alpha_B : float
        Recombination coefficient for case B "on the spot" recombination.
    debug : bool
        Debug flag to enable checks for physical result.

    Returns
    -------
    float
        The plus solution. Checks if is physical, in the interval [0, 1].
    """
    A = (n_H*alpha_B)/gamma
    A = np.maximum(A, 1e-5)
    X = (np.sqrt(1 + 4*A) - 1) / (2*A)

    if debug:
        if 0 < X and X < 1:
            return X
        else:
            raise(BaseException("Non physical solution! Value obtained: " + str(X)))
    
    return X