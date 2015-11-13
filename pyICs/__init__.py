"""
    Copyright (C) 2015 Jakob Herpich (herpich@mpia.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""

from . import am_profiles, density_profiles, equilibrium_halos, tools
import scipy.interpolate as interp
import numpy as np

def create_ics(**kwargs):
    """
    A convenience function which basically does everything for you. You only need to define
    the density profile and its first and second derivative with respect to radius. You can
    also use a class of predefined density profiles including Hernquist, NFW and Jaffe
    profiles or isothermal spheres i.e. the alpha-beta-gamma profiles with the respective
    parameters.

    As of now the code only supports the gas and dark matter density to share the same radial
    dependence with a spatially constant ratio of gas to dark matter density.

    There are a number of parameters you get to set. The most important ones are (default
    values in parantheses)

    Virial mass: m_vir ('1e12 Msol')

    Gas halo spin: spin_parameter (0.04)

    Dictionary of parameters which are passed as an argument to the density profile function:
        pars ({'alpha': 1., 'beta': 3., 'gamma': 1., 'c': 10., 'factor': 0.1})

        These are the parameters for the alpha-beta-gamma models. 'c' is the halo
        concentration in these models and 'factor' is the relative scale outside the virial
        radius at which halos with beta <= 3 are smoothly truncated to ensure a finite
        halo mass.

    Particle number (DM and gas respectively): n_particles (1e5)

    Gas fraction of the halo: f_bary (0.1)

    Output file name: f_name ('halo.out')
    
    Examples:
    ---------

    Hernquist:
    >>> from pyICs.density_profiles import *
    >>> pars = {'alpha': 1., 'beta': 4., 'gamma': 1.}
    >>> sim = create_ics(profile=alphabetagamma, drhodr=dalphabetagammadr,
    >>>     d2rhodr2=d2alphabetagammadr2, pars=pars, m_vir='1e12 Msol', n_particles=1e6)
    >>> sim
    <SimSnap "<created>" len=1000000>

    Arbitrary:
    ----------
    First the density profile and its first and second derivative with respect to the
    dimensionless radius x = r/R_s need to be defined analytically. They need to
    accept the two argumets 'x' and 'pars'. If pars is not needed, it can just be set
    to 'None' but the function definition must accept a 'pars' argument.
    >>> def rho(x, pars):
        ...
        return rho_of_x
    >>> def drhodr(x, pars):
        ...
        return drhodr_of_x
    >>> def d2rhodr2(x, pars):
        ...
        return d2rhodr2_of_x

    Then define specific parameters for that function (can be set to 'None').
    >>> pars = ...
    >>> myhalo = create_ics(profile=rho, drhodr=drhodr, d2rhodr2=d2rhodr2, pars=pars)

    """

    args = {}
    m_vir = kwargs.get('m_vir', '1e12 Msol')
    args['m_vir'] = pynbody.units.Unit(m_vir)
    args['h'] = kwargs.get('h', 0.7)
    args['overden'] = kwargs.get('overden', 200)
    args['pars'] = kwargs.get('pars', {'alpha': 1., 'beta': 3., 'gamma': 1.,
        'c': 10., 'factor': 0.1})
    profile = kwargs.get('profile', density_profiles.alphabetagamma)
    if profile == density_profiles.alphabetagamma:
        if args['pars']['beta'] <= 3. and 'factor' not in args['pars'].keys():
            args['pars']['factor'] = 0.1
        if 'c' not in args['pars'].keys():
            args['pars']['c'] = 10.
    else:
        unit = kwargs.get('length_unit', '1 kpc')
        r_vir = tools.calc_r_vir(args['m_vir'], args['h'], args['overden'])
        args['c'] = r_vir.in_units(unit)
    kwargs.update(args)
    halo = equilibrium_halos.EquilibriumHalo(**kwargs)
    halo.make_halo()
    halo.finalize()
    return halo.sim
