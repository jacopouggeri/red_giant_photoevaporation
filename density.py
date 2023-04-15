import numpy as np
import astropy.units as u # manages units
from scipy.special import lambertw # Lambert W function

from astropy.constants import G

def density_hydrostatic(R, R_c, R_s, rho_s):
  """ Find density profile in hydrostatic isothermal equilibrium.  The temperature dependence is encoded in
   the value of r_c, which is calculated from the speed of sound in the hydrostatic region.

    Parameters
    ----------
    R : float
        Radial distance
    R_c : float
        Parker wind critical radius
    R_s: float
        Radius of star (photospere)
    rho_s: float
        Density at the photosphere of the star

    Returns
    -------
    float
        Density at radius R for a star in hydrostatic equilibrium with density rho_s at radius R_s.
    """
  arg = 1/R - 1/R_s
  return rho_s*np.exp(2*R_c*arg)
    
def mach_parker(r):
    """ Find mach number of steady parker wind at dimensionless radial position r.

    Parameters
    ----------
    r : float
        Dimensionless radial distance R/R_c where R_c is the critical radius of the Parker wind and R is the radial distance

    Returns
    -------
    float
        Mach number (v/c_s) at position r. Adimensional.
    """
    ratio = 1/r
    arg = -(ratio)**4 * np.exp(3 - 4*ratio)
    # Use branch 0 for r < 1 and branch -1 for r > 1
    out = np.sqrt(np.real(-lambertw(arg, 0))) * np.heaviside(ratio - 1, 1)
    out += np.sqrt(np.real(-lambertw(arg, -1))) * np.heaviside(1 - ratio, 1)
    return out

def density_parker(r, rho_c):
    """ Density profile with dimensionless radial distance for steady parker wind

    Parameters
    ----------
    r : float
        Dimensionless radial distance R/R_c where R_c is the critical radius of the Parker wind and R is the radial distance
    rho_c : float
        Density at the critical radius

    Returns
    -------
    float
       The density as a function of the radial distance r for a steady parker wind solution.
    """
    M = mach_parker(r)
    return (1/r)**2 * rho_c/M

def density_parker_ifront(r, r_I, rho_p):
    """ Density profile with dimensional radial distance for steady parker wind, knowing the density rho_p at position \
        r_I of the ionisation front

    Parameters
    ----------
    r : float
        Dimensionless radial distance R/R_c where R_c is the critical radius of the Parker wind and R is the radial distance
    r_I : 
        Dimensionless radial position of the ionisation front R_I/R_c
    rho_p : float
        Density at the ionisation front

    Returns
    -------
    float
       The density as a function of the radial distance R for a steady parker wind solution with density rho_p at R_I.
    """
    m_ratio = mach_parker(r_I)/mach_parker(r)
    return ((r_I/r)**2) * rho_p * m_ratio

def momentum_balance_density(r_I, rho_H, T_ratio):
    """ Find density in the parker wind region at the ionization front r_I, applying momentum balance.

    Parameters
    ----------
    r_I : 
        Dimensionless radial position of the ionisation front R_I/R_c
    rho_H: float
        Density at r_I in the hydrostatic region.
    T_ratio: float
        Temperature ratio of the two regions T_H/T_P

    Returns
    -------
    float
        Density at r_I for parker wind density profile
    """
    rho_P = (1/(1 + mach_parker(r_I)**2))*T_ratio*rho_H
    return rho_P

def parker_critical(c_s, M_s):
    """Parker wind critical radius for speed of sound c_s for a star of mass M_s

    Parameters
    ----------
    c_s : float
        speed of sound in the parker wind
    M_s : float
        mass of the star

    Returns
    -------
    float
        The parker wind critical radiuss
    """
    return G*M_s/(2*c_s**2)

def density_profile(r, r_I, R_s, M_s, rho_s, a_H, a_P):
    """ Return correct parker wind density profile valid for r > r_I. 
    Calculated by using the density at the ionisation front, found by applying momentum balance at the ionisation front
    under the assumption that the region between the photosphere and the ionisation front is hydrostatic. 

    Parameters
    ----------
    r : float
        Dimensionless radial distance R/R_c (R_c is the parker critical radius)
    r_I : float
        Dimensionless position of the ionisation front R_I/R_c
    R_s : float
        Radius of the star (at the photosphere)
    M_s : float
        Mass of the star
    rho_s : float
        Density at R_s
    a_H : float
        Speed of sound in the hydrostatic region (R_s/R_c < r < r_I)
    a_P : float
        Speed of sound in the parker wind region (r > r_I)

    Returns
    -------
    float
        Density at position r
    """
    # Calculate hydrostatic density at R_I
    R_c_hydrostatic = parker_critical(a_H, M_s)
    R_c = parker_critical(a_P, M_s)
    R_I = R_c*r_I
    rho_H = density_hydrostatic(R_I, R_c_hydrostatic, R_s, rho_s)

    # Calculate parker density at R_I
    T_ratio = (a_H/a_P)**2
    rho_P = momentum_balance_density(r_I, rho_H, T_ratio)

    # Return parker density profile
    return density_parker_ifront(r, r_I, rho_P)

def full_density_profile(r, r_I, R_s, M_s, rho_s, a_H, a_P):
    """Return full density profile outside the star

    Parameters
    ----------
    r : float
        Dimensionless radial distance R/R_c (R_c is the parker critical radius)
    r_I : float
        Dimensionless position of the ionisation front R_I/R_c
    R_s : float
        Radius of the star (at the photosphere)
    M_s : float
        Mass of the star
    rho_s : float
        Density at R_s
    a_H : float
        Speed of sound in the hydrostatic region (R_s/R_c < r < r_I)
    a_P : float
        Speed of sound in the parker wind region (r > r_I)

    Returns
    -------
    float
        Full density profile valid from r > R_s/R_c to infinity.
    """
    rhop = density_profile(r, r_I, R_s, M_s, rho_s, a_H, a_P)
    rhoh = density_hydrostatic(r*parker_critical(a_P, M_s), parker_critical(a_H, M_s), R_s, rho_s)
    # Hydrostatic density profile in the hydrostatic region and parker profile in the parker wind region
    return rhop*np.heaviside(r - r_I, 1) + rhoh*np.heaviside(r_I - r, 1)