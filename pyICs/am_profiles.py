"""

am_profiles
================

This is a simple collection of angular momentum profiles
m = M(<j)/M_vir/f_bary.
They will be used to calculate velocities of the gas particles.
They have to be defined as follows:

>>> def profile(j, pars):
>>>    <Funciton definition>
>>>    m = ...
>>>    return m

"""

import numpy as np

def bullock_prof(j, pars):
    """

    This function returns the normalized cumulative mass as a function
    of specific angular momentum j as defined in Bullock et. al (MNRAS 2001, 396, 121-140)
    j must be normalized to the maximum value inside r_vir, i.e. j must be in [0,1].

    Parameters:
    -----------

    j: specific angular momentum array

    pars: Dictionary containing the shape parameter mu
        >>> pars = {'mu': 1.3}

    Returns:
    --------
    m(<j): Angular momentum profile

    """

    return mu*j/(j+mu-1.)
