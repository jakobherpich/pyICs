"""

Tools
=====

Simple numerical toolset for sampling dark matter halos

"""

import numpy as np

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
