import numpy as np
import matplotlib.pyplot as plt # for plotting
import astropy.units as u # manages units
from scipy import integrate
from scipy.optimize import brentq

from ionisation_recombination import sigma_nu, ionisation_fraction
from EUV_flux import disk_spectrum
from density import parker_critical, density_profile

# Constants
from astropy.constants import M_sun, R as R_gas, G, m_p, h
k = 1e-2 * u.cm**2/u.g # opacity
alpha_B = 2.6e-13 * (u.cm **3)/u.s # cm3s-1 at (T = 10^4 K) From Owen and Alvarez 2016

def scale_height(c, M, R):
    return c**2/(G*M/R**2)

def surface_density(c, M, R):
    k = 1e-2 * u.cm**2/u.g # opacity
    H = scale_height(c, M, R)
    return 1/(k*H)

def ionisation_fraction_error(r_I, M_s, M_BH, m_dot, R_s, a_H, a_P, d, R_bins = 300, nu_bins = 1000, r_size = 10):
    """ Calculates numerically the ionisation fraction around a star, for a given guess of the ionisation front position r_I. 
    Then returns 

    Parameters
    ----------
    r_I : float
        Starting guess for ionisation front position in dimensionless units R_I/R_c where R_c is the parker critical radius.
    M_s : float
        Mass of star.
    M_BH : float
        Mass of black hole.
    m_dot : float
        Accretion rate of black hole in fractions of Eddington accretion rate. Accepts value from 0 to 1.
    R_s : float
        Radius of the star.
    a_H : float
        Speed of sound in the hydrostatic region (R < R_I)
    a_P : float
        Speed of sound in parker region (R > R_I)
    d : float
        Distance of star from black hole.
    R_bins : int, optional
        Number of radial bins for integration, by default 300
    nu_bins : int, optional
        Number of frequency bins for integration, by default 1000

    Returns
    -------
    float
        The difference between the calculated ionisation fraction at r_I and a small acceptance threshold value of 10^-4.
    """
    R_c = parker_critical(a_P, M_s)
    R_I = r_I * R_c # Need position in meters for some calculations
    r = np.logspace(np.log10(r_I), np.log10(r_size), R_bins) # Set radial bins
    r = r[::-1] # Invert radial bins
    R = R_c*r # Radial bins in multiples of R_c

    rho_s = surface_density(a_H, M_s, R_s)

    # Use density profile to find the number density
    rho = density_profile(r, r_I, R_s, M_s, rho_s, a_H, a_P)
    n_H = (rho/m_p).to("m-3") # use proton mass as hydrogen mass (error is negligible)
    
    # Find accretion disk flux
    L_nu, nu = disk_spectrum(m_dot, M_BH, nu_bins)
    
    # Set arrays that store necessary variables for integration
    dgamma_0 = sigma_nu(nu) * L_nu/(h*nu) * 1/(4*np.pi*d**2) # Set initial values for convenience
     # Integrand to find gamma (has a radial and frequency index as frequency will be integrated)
    dgamma = np.ones((R_bins, nu_bins)) * np.transpose(dgamma_0) 
    gamma = np.ones(R_bins) / u.s # Recombination rate
    X = np.ones(R_bins) # Ionisation fraction
    tau = np.ones((R_bins, nu_bins)) # optical depth

    for i in range(len(R) - 1): # Loop to integrate radially
        dtau = (n_H[i]*sigma_nu(nu).transpose()*(1 - X[i])*(R[i] - R[i+1])).decompose()
        tau[i+1] = tau[i] + dtau # extend optical depth
        dgamma[i+1] *= np.exp(-tau[i+1]) # apply optical depth decay to incoming flux
        gamma[i+1] = integrate.trapezoid(dgamma[i+1], nu.to("Hz")) # Integrate over frequencies to find total flux
        X[i+1] = ionisation_fraction(gamma[i+1], n_H[i], alpha_B, debug = False)

    return X[-1] - 1e-4

def ionisation_fraction_plot(r_I, M_s, M_BH, m_dot, R_s, a_H, a_P, d, R_bins = 300, nu_bins = 1000):
    R_c = parker_critical(a_P, M_s)
    R_I = r_I * R_c
    r = np.logspace(np.log10(r_I), 1, R_bins)
    r = r[::-1] # invert radial bins
    R = R_c*r # radial bins in multiples of R_c

    rho_s = surface_density(a_H, M_s, R_s)

    
    # Use density profile to find the number density
    rho = density_profile(r, r_I, R_s, M_s, rho_s, a_H, a_P)
    n_H = (rho/m_p).to("m-3") # use proton mass as hydrogen mass
    
    L_nu, nu = disk_spectrum(m_dot, M_BH, nu_bins)
    
    dgamma_0 = sigma_nu(nu) * L_nu/(h*nu) * 1/(4*np.pi*d**2)
    dgamma = np.ones((R_bins, nu_bins)) * np.transpose(dgamma_0)
    gamma = np.ones(R_bins) / u.s
    X = np.ones(R_bins)
    A = np.zeros(R_bins)
    tau = np.ones((R_bins, nu_bins))

    for i in range(len(R) - 1):
        dtau = (n_H[i]*sigma_nu(nu).transpose()*(1 - X[i])*(R[i] - R[i+1])).decompose()
        tau[i+1] = tau[i] + dtau
        dgamma[i+1] *= np.exp(-tau[i+1])
        gamma[i+1] = integrate.trapezoid(dgamma[i+1], nu.to("Hz"))
        A[i+1] = n_H[i]*alpha_B/gamma[i+1]
        X[i+1] = ionisation_fraction(gamma[i+1], n_H[i], alpha_B, debug = False)

    return r, X
"""
def generate_guess_dense(args):
    guess_1 = 0.2
    guess_2 = 0.1
    for i in range(30):
        sign1 = np.sign(ionisation_fraction_error(guess_1, *args))
        sign2 = np.sign(ionisation_fraction_error(guess_2, *args))
        if sign1 == sign2:
            if sign1 > 0:
                guess_2 -= 0.05
                guess_1 -= 0.01
            elif sign1 < 0:
                guess_1 += 0.05
                guess_2 += 0.01
        elif sign1 <= sign2:
            guess_1 -= 0.05
            guess_2 -= 0.05
        elif sign1 >= sign2:
            break
    return guess_1, guess_2

def generate_guess_light(args):
    guess_1 = 3
    guess_2 = 1.5
    for i in range(30):
        sign1 = np.sign(ionisation_fraction_error(guess_1, *args))
        sign2 = np.sign(ionisation_fraction_error(guess_2, *args))
        if sign1 == sign2:
            if sign1 > 0:
                guess_2 -= 0.5
            elif sign1 < 0:
                guess_1 += 0.5
        elif sign1 != sign2:
            break
    return guess_1, guess_2
"""

def update_guess(guess_1, guess_2, step, args):
    R_s = args[3]
    R_c = parker_critical(args[6], args[0])
    r_s = (R_s/R_c).decompose().value
    sign1 = np.sign(ionisation_fraction_error(guess_1, *args))
    sign2 = np.sign(ionisation_fraction_error(guess_2, *args))
    for i in range(10):
        if sign1 == sign2:
            if sign1 > 0:
                guess_2 -= step
                sign2 = np.sign(ionisation_fraction_error(guess_2, *args))
            elif sign1 < 0:
                guess_1 += step
                sign1 = np.sign(ionisation_fraction_error(guess_1, *args))
        elif sign1 < sign2:
            guess_1 = r_s + step
            guess_2 = r_s
            sign1 = np.sign(ionisation_fraction_error(guess_1, *args))
            sign2 = np.sign(ionisation_fraction_error(guess_2, *args))
        elif sign1 > sign2:
            break
    return guess_1, guess_2, sign1, sign2

def generate_guess_dense(args):
    R_s = args[3]
    R_c = parker_critical(args[5], args[0])
    r_s = (R_s/R_c).decompose().value
    step = 0.1
    for j in range(10):
        guess_1 = r_s + step
        guess_2 = r_s
        guess_1, guess_2, sign1, sign2 = update_guess(guess_1, guess_2, step, args)
        if sign1 <= sign2:
            step *= 0.5
        elif sign1 > sign2:
            break
    return guess_1, guess_2

def generate_guess_light(args):
    R_s = args[3]
    R_c = parker_critical(args[5], args[0])
    r_s = (R_s/R_c).decompose().value
    step = 0.5
    for j in range(10):
        guess_1 = r_s + step
        guess_2 = r_s 
        guess_1, guess_2, sign1, sign2 = update_guess(guess_1, guess_2, step, args)
        if sign1 <= sign2:
            step *= 0.5
        elif sign1 > sign2:
            break
    return guess_1, guess_2

def light_or_dense(M_s, R_s):
    coeff = [0.21663171, 0.00581934]
    threshold = np.poly1d(coeff)
    return (R_s.value < threshold(M_s.value))

def ionisation_front(m_dot, M_s, M_BH, R_s, a_H, a_P, d, R_bins = 300, debug = False, r_size = 10):
    """Uses root finding algorithm brentq applied to ionisation_fraction_error
    function to find position of the ionisation front for given parameters.

    Parameters
    ----------
    m_dot : float
        Accretion rate of black hole in fractions of Eddington accretion rate. Accepts value from 0 to 1.
    M_s : float
        Mass of star.
    M_BH : float
        Mass of black hole.
    R_s : float
        Radius of the star.
    a_H : float
        Speed of sound in the hydrostatic region (R < R_I)
    a_P : float
        Speed of sound in parker region (R > R_I)
    d : float
        Distance of star from black hole.
    debug : bool, optional
        Debug option, by default False

    Returns
    -------
    float
        Position of the ionisation front in dimensionless units R_I/R_c where R_c is the parker critical radius.
    """
    args = (M_s, M_BH, m_dot, R_s, a_H, a_P, d, R_bins, r_size)
    
    if light_or_dense(M_s, R_s):
        guess_1, guess_2 = generate_guess_dense(args)
        if debug: 
            print("dense", guess_1, guess_2)
    else:
        guess_1, guess_2 = generate_guess_light(args)
        if debug: 
            print("light", guess_1, guess_2)
    
    r_I = brentq(ionisation_fraction_error, guess_1, guess_2, args, full_output=debug)
    if not debug:
        return r_I
    else:
        print(r_I)
        return r_I[0]
    

