"""

density_profiles
================

This is a simple collection of common density profiles to be sampled.
They will be used to create isolated equilibrium halos.
They have to be defined as follows:

>>> def profile(x, pars):
>>>    <Funciton definition>
>>>    rho_of_x = ...
>>>    return rho_of_x

x has to be an numpy array in units of some scale radius r_s and the
return value is a numpy array storing the density as a function of x
in arbitrary units.

"""

import numpy as np

def alphabetagamma(x, pars):
    """

    Simple implementation of (alpha, beta, gamma) models (2004ApJ...601..37)
    with an exponential cutoff for cases at which the halo mass would not be
    finite otherwise (i.e. beta < 3):

    rho(x) = 2. / (x^gamma * (1 + x^alpha)^((beta-gamma)/alpha))

    Examples:
    ---------

    >>> pars = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'c': c, 'factor': factor}
    >>> x = np.logspace(-3, 3, n_sample)
    >>> rho = alphabetagamma(x, pars)

    Isothermal: alpha = arbitrary, beta = 2., gamma = 2.
    NFW: alpha = 1., beta = 3., gamma = 1.
    Hernquist: alpha = 1., beta = 4., gamma = 1.
    Jaffe: alpha = 1., beta = 4., gamma = 2.

    Parameters:
    -----------

    x: Position array in units of scale radius (R_s). Must be monotonically increasing.

    pars: Dictionary with profile parameters which must contain the following:
        alpha: Parameter that controls width of the transition between inner and
            outer slope
        beta: Outer slope
        gamma: Inner slope
        c (only necessarry if b <= 3): Concentration parameter (R_vir/R_s)
        factor (only necessarry if b <= 3): Scale length for exponential cutoff
            in units of R_vir

    Returns:
    --------
    rho(x): Density as a function of radius

    """

    alpha = pars['alpha']
    beta = pars['beta']
    gamma = pars['gamma']
    rho = 2./((x**gamma)*((1+x**alpha)**((beta-gamma)/alpha)))
    if beta <= 3.:
        c = pars['c']
        factor = pars['factor']
        outer = np.where(x>c)
        eps = -(gamma+beta*(c**alpha))/(1.+c**alpha) + 1./factor
        rho[outer] = 2.*np.exp(-(x[outer]/c-1.)/factor)*((x[outer]/c)**eps)
        rho[outer] /= (c**gamma)*((1+c**alpha)**((beta-gamma)/alpha))
    return rho

def dalphabetagammadr(x, pars):
    """

    Funciton that returns the first derivative of the alphabetagamma density
    models with respect to radius in units of scale radius R_s

    Parameters:
    -----------

    x: Radial array in units of R_s

    pars: Dictionary with profile parameters (see docstring of alphabetagamma() for details)

    Returns:
    --------

    drhodr(x): First derivative of density
    """
    alpha = pars['alpha']
    beta = pars['beta']
    gamma = pars['gamma']
    fac = -(gamma + (beta-gamma)/(1.+x**(-alpha)))/x
    if beta <=3.:
        c = pars['c']
        factor = pars['factor']
        outer = np.where(x>c)
        eps = -(gamma+beta*(c**alpha))/(1.+c**alpha) + 1./factor
        fac[outer] = eps/x[outer] - 1./c/factor
    return fac*alphabetagamma(x, pars)

def d2alphabetagammadr2(x, pars):
    """

    Funciton that returns the second derivative of the alphabetagamma density
    models with respect to radius in units of scale radius R_s

    Parameters:
    -----------

    x: Radial array in units of R_s

    pars: Dictionary with profile parameters (see docstring of alphabetagamma() for details)

    Returns:
    --------

    d2rhodr2(x): Second derivative of density
    """
    alpha = pars['alpha']
    beta = pars['beta']
    gamma = pars['gamma']
    denom = (1.+x**(-alpha))
    fac = gamma + (beta-gamma)/denom + (gamma-beta)/denom/denom/(x**alpha)
    fac += (gamma + (beta-gamma)/denom)**2
    fac /= x*x
    if beta <=3.:
        c = pars['c']
        factor = pars['factor']
        outer = np.where(x>c)
        eps = -(gamma+beta*(c**alpha))/(1.+c**alpha) + 1./factor
        fac[outer] = (eps/x[outer] - 1./c/factor)**2 - eps/x[outer]/x[outer]
    return fac*alphabetagamma(x, pars)
