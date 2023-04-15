# Modules
import scipy as sp
import numpy as np
import scipy.integrate as integrate # for numerical integration
from astropy import units # manages units

# Import constant with units from the astropy module
# https://docs.astropy.org/en/stable/constants/index.html#collections-of-constants-and-prior-versions
from astropy.constants import M_sun, m_p, sigma_T, sigma_sb, b_wien, G, c, h, k_B, L_sun, R_sun 

# Ignore divide by zero errors (will generate NaNs but those will be replaced by zeros)
np.seterr(divide='ignore')

# Define functions

def T_eff(r, T_s):
    """ Calculate the effective temperature profile in Kelvin of an accretion disk with temperature scale factor T_s

    Parameters
    ----------
    r : float
        Dimensionless radial distance (R/R_in)
    T_s : float
        Temperature scale factor in Kelvin

    Returns
    -------
    float
        Effective temperature in Kelvin
    """
    return T_s * (r**(-3) - r**(-7/2))**(1/4)

def planck_law(nu, T):
    """ Returns the power emitted per unit area of a blackbody at temperature T,
    into a unit solid angle, in the frequency interval nu to nu + d(nu).
    Requires Boltzmann constant k_B from the astropy module.
    
    Parameters
    ----------
    nu : float
        Frequency in Hz
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Specific intensity in Ws/m2 (per solid angle!)
    """
    a = h*nu/(k_B*T) # dimensionless argument of the exponential
    exp_term = (c**2)*(np.exp(a) - 1)
    return 2*h*(nu**3)/exp_term

def planck_law_radial(nu, r, T_s):
    """Returns the power emitted per unit area of a passive accretion disk with temperature profile T_eff(r, T_s),
    into a unit solid angle, in the frequency interval nu to nu + d(nu), at dimensionless radius r.

    Parameters
    ----------
    nu : float
        Frequency in Hz
    r : float
        Dimensionless radial distance (R/R_in)
    T_s : float
        Temperature scale factor in Kelvin

    Returns
    -------
    float
        Specific intensity in Ws/m2 (per solid angle!)
    """
    return planck_law(nu, T_eff(r, T_s))

def disk_spectrum(m_dot, M, nu_bins=1000, r_in=3, nu_start = 13.6):
    """Calculate accretion disk spectrum for a black hole of mass M and an accretion disk of inner
    radius r_in*R_s where R_s is the Schwarschild radius of the black hole

    Parameters
    ----------
    m_dot : float
        Should be between 0 and 1, ratio of the disk accretion rate to the Eddington rate
    M : float
        Mass of the black hole in kg
    r_in : int, optional
        Ratio of the inner radius of the accretion disk to the black hole Schwarschild radius, by default 3
    nu_bins: int, optional
        Number of bins in nu, by default 1000

    Returns
    -------
    tuple(numpy.ndarray(float64), numpy.ndarray(float64))
        The frequency spectrum of the accretion disk and an array of nu_bins (default 1000) frequency values for plotting and integrating.
        Both are logarithmically spaced.
    """
    # Calculate necessary quantities
    R_s = 2*G*M/(c**2) # Schwarschild radius 
    R_in = r_in*R_s # innermost stable orbit (approximate)
    M_dot_edd = 8*np.pi*R_in*m_p*c/sigma_T # accretion rate at Eddington limit
    M_dot = m_dot*M_dot_edd # chosen fraction of Eddington accretion rate
    T_s = (3*G*M*M_dot/(8*np.pi*sigma_sb*(R_in**3)))**(1./4.) # temperature scale factor for temperature distribution

    # Prepare input arrays
    hnu = np.logspace(start=np.log10(nu_start), stop=3, num=nu_bins) * units.eV # photon energy values array
    nu = hnu/h # frequency values array (useful for integration)
    r = np.logspace(start=0, stop=4, num=1000)
    r_mesh, nu_mesh = np.meshgrid(r, nu) # first r then nu to integrate over r first

    specific_intensity = planck_law_radial(nu_mesh, r_mesh, T_s)

    # Catch and remove NaNs
    nan_location = np.isnan(specific_intensity) 
    specific_intensity[nan_location] = 0 # remove NaN values

    # Prepare integrand
    integrand = 2 * np.pi**2 * R_in**2 * r_mesh * specific_intensity # 2 pi^2 R_in^2 r B_nu(r)
    integrand = integrand.to("W*s")

    # Integrate to find acctretion disk spectrum
    # units are stripped using .value then put back in place manually
    temp = []
    for r_slice in integrand.value:
        temp.append(integrate.trapezoid(r_slice, r))
    L_nu = np.array(temp) * units.W * units.s
    return L_nu, nu


def euv_flux_watt(m_dot, M, r_in=3):
    """Calculate EUV flux (hnu > 13.6 eV) from the accretion disk of a black hole of mass M accreting
     at an accretion rate m_dot fractions of its Eddington accrettion rate

    Parameters
    ----------
    m_dot : float
        Should be between 0 and 1, ratio of the disk accretion rate to the Eddington rate
    M : float
        Mass of the black hole in kg
    r_in : int, optional
        Ratio of the inner radius of the accretion disk to the black hole Schwarschild radius, by default 3

    Returns
    -------
    float
        EUV flux in Watt
    """
    # Calculate accretion disk spectrum
    L_nu, nu = disk_spectrum(m_dot, M)

    # Integrate the spectrum to find the luminosity or energy flux
    # units are stripped using .value then put back in place manually
    flux = integrate.trapezoid(L_nu.value, nu.to("Hz").value) * units.W
    return flux

def euv_flux_photons(m_dot, M, r_in = 3):
    """Calculate EUV photon flux (hnu > 13.6 eV) from the accretion disk of a black hole of mass M accreting
     at an accretion rate m_dot fractions of its Eddington accrettion rate

    Parameters
    ----------
    m_dot : float
        Should be between 0 and 1, ratio of the disk accretion rate to the Eddington rate
    M : float
        Mass of the black hole in kg
    r_in : int, optional
        Ratio of the inner radius of the accretion disk to the black hole Schwarschild radius, by default 3

    Returns
    -------
   float
        EUV photon flux in photon/s
    """
    # Calculate accretion disk spectrum
    L_nu, nu = disk_spectrum(m_dot, M)

    # Find pPhoton flux in frequency range nu to nu +  d(nu)
    photon_flux_nu = L_nu/(h*nu) 

    # Integrate the photon flux spectrum to find the total photon flux
    # units are stripped using .value then put back in place manually
    photon_flux = integrate.trapezoid(photon_flux_nu.to("").value, nu.to("Hz").value) * units.Hz
    return photon_flux