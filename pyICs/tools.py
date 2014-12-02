"""

Tools
=====

Simple numerical toolset for sampling dark matter halos

"""

import numpy as np
from pynbody import array, units

def simpsons_integral(x, integrand_of_x, zero=False, norm_ind=0):
    """
    Calculate the integral I[i]=int_x[0]^x[i] integrand(x)dx
    as a function of the input array according to Simpson's rule.
    If zero is set to 'True' the lower bound is taken to be zero and thus
    I[0] = int_0^x[0] integrand(x)dx.

    Parameters:
    -----------

    x : array

    integrand_of_x : function which calculates integrand(x) as a function of x
    (it is likely to use the lambda keyword here)

    zero (True) : if 'True' lower bound is taken to be zero

    norm_ind (0) : This parameter sets the normalization of the integral, i.e. a
        constant is added to the return array such that I[norm_ind] = 0. This is useful
        for calculating integrals that are set to be 0 at x=infty.

    """

    dx = x[1:]-x[:-1]
    sum1 = integrand_of_x(x[:-1])+integrand_of_x(x[1:])
    sum2 = integrand_of_x(0.5*(x[:-1]+x[1:]))
    if zero:
        first = x[:1]*(4.*integrand_of_x(x[:1]/2.)+integrand_of_x(x[:1]))/6.
    else:
        first = np.array([0.])
    s = np.append(first, (dx*(sum1+4.*sum2)/6.)).cumsum()
    if not zero:
        return s - s[norm_ind]
    else:
        return s

def smooth_exponential_cutoff(x):
    ret = (2. - 3./np.e)*x**3 + (4./np.e - 3.)*x**2 + 1.
    return ret

def outer_smooth_cutoff(x, scale):
    ret = smooth_exponential_cutoff((x-1.)/scale)
    inner = np.where(x < 1.)
    ret[inner] = np.ones(x[inner].shape)
    outer = np.where(x >= 1.+scale)
    ret[outer] = np.exp((1.-x[outer])/scale)
    return ret

def calc_rho_crit(h):
    """Calculate critical density for given Hubble constant"""
    return 3.*h**2*units.Unit('1e2 km s**-1 Mpc**-1')**2/8./np.pi/units.G

def calc_r_vir(m_vir, h, overden):
    """Calculate virial radius (w.r.t. rho_crit)"""
    rvir3 = 3./4./np.pi/overden*m_vir/calc_rho_crit(h)
    return units.Unit('{0:.64g} kpc'.format(float(rvir3.in_units('kpc**3')**(1./3.))))

def iterate_temp(gas, tol=1e-4):
    """
    Solve for the equilibrium temperature iteratively, given an initial pressure
    and the corresponding temperature assuming a mean molecular weight of unity.
    At each step the temperature is varied, the mean molecular mass recalculated
    according to the new temperature and the given pressure.

    The resulting temperatures are directly stored into the input snapshot.

    Parameters:
    -----------

    gas : pynbody snapshot of gas particles, must have arrays 'temp' and 'pressure'

    Keyword argumens:
    -----------------

    tol (1e-4) : relative tolerance level at which temperature is considered to
                 be converged

    """
    temp = gas['temp'].in_units('K').view(np.ndarray)
    pressure = gas['pressure'].in_units('g cm**-1 s**-2').view(np.ndarray)

    t_diff = np.ones(temp.shape)
    t_orig = temp
    t_old = temp
    mu = 1.
    mu = get_mu(mu*temp, pressure)
    mu_high = np.ones(mu.shape)
    mu_low = np.ones(mu.shape)
    high = np.where(mu > mu_high)
    low = np.where(mu < mu_low)
    mu_high[high] = mu[high]
    mu_low[low] = mu[low]
    mu_half = (mu_high+mu_low)/2.
    while ((np.abs(mu_high-mu_low)/mu_low) > tol).any():
        mu = get_mu(mu_half*temp, pressure)
        high = np.where(mu_half*temp, pres)
        mu_low[high] = mu_half[high]
        low = np.where(mu <= mu_half)
        mu_high[low] = mu_half[low]
        mu_half = (mu_high+mu_low)/2.
    gas['temp'] = array.SimArray(mu_half*temp, 'K')

def get_mu(T, elecPres) : 
    """
    Calculate mean molecular weight given the temperature and pressure
    using hydrogen and helium.


    Parameters:
    -----------

    T : temperature in K

    elecPres: initial electron pressure

    """
    
    HIIionE =13.5984*1.60217733e-12
    HeIIionE =24.5874*1.60217733e-12
    HeIIIionE =54.417760*1.60217733e-12
    Htot =0.9
    Hetot =0.1

    fracHIIHI = calc_saha(T, 1., 2., elecPres, HIIionE);
    HII = fracHIIHI/(fracHIIHI+1.);
    HI = 1./(fracHIIHI+1.);
    fracHe2n3HeI = calc_saha(T, 3., 4., elecPres, HeIIionE);
    He2n3 = fracHe2n3HeI/(fracHe2n3HeI+1.);
    HeI = 1./(fracHe2n3HeI+1.);
    fracHeIIIHeII = calc_saha(T, 2., 3., elecPres, HeIIIionE);
    HeIII = He2n3*fracHeIIIHeII/(fracHeIIIHeII+1.);
    HeII = He2n3/(fracHeIIIHeII+1.);
    return HII*Htot/2.+HI*Htot+4.*Hetot*(HeIII/3.+HeII/2.+HeI);


def calc_saha(T, partTop, partBottom, elecPres, IonEnergy) : 
    kB = 1.38066e-16 # erg kelvin^-1 
    TwoPimekBonhh = 17998807946.6
    return 2.*kB*T/elecPres*partTop/partBottom*(TwoPimekBonhh*T)**1.5*np.exp(-IonEnergy/kB/T)
