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



equilibrium_halos
=================

"""

import numpy as np
from scipy import interpolate as interp
import sys
import time
from . import am_profiles, density_profiles, tools
from pynbody import array, new, snapshot, units
from pynbody import gadget, grafic, nchilada, ramses, tipsy

class SampleDarkHalo:
    """

    SampleDarkHalo
    ==============

    A class that is initialized with a set of halo parameters (e.g. mass, concentration,
    density profile etc.) and creates an equilibrium spherical non-rotating dark matter
    halo which is created by sampling particle velocities from the distribution function
    as described by B&T ch. 4.3.1. and Kazantzidis et al. 2004, ApJ, 601, 37

    The default parameters describe an NFW halo with mass 1e12 Msol and 1e5 particles.

    Examples:
    ---------

    Hernquist:
    >>> from pyICs.density_profiles import *
    >>> pars = {'alpha': 1., 'beta': 4., 'gamma': 1.}
    >>> hern = SampleDarkHalo(profile=alphabetagamma, drhodr=dalphabetagammadr,
    >>>     d2rhodr2=d2alphabetagammadr2, pars=pars, m_vir='1e12 Msol', n_particles=1e6)
    >>> hern.sample_equilibrium_halo()
    >>> sim = hern.sim
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
    >>> myhalo = SampleDarkHalo(profile=rho, drhodr=drhodr, d2rhodr2=d2rhodr2, pars=pars)
    """

    def __init__(self, **kwargs):
        self.__profile = kwargs.get('profile', density_profiles.alphabetagamma)
        self.__drhodr = kwargs.get('drhodr', density_profiles.dalphabetagammadr)
        self.__d2rhodr2 = kwargs.get('d2rhodr2', density_profiles.d2alphabetagammadr2)
        self.__pars = kwargs.get('pars', {'alpha': 1., 'beta': 3., 'gamma': 1.,
            'c': 10., 'factor': 0.1})
        if self.__profile == density_profiles.alphabetagamma and self.__pars['beta'] <= 3.:
            if 'factor' not in self.__pars.keys(): self.__pars['factor'] = 0.1
        self.__m_vir = kwargs.get('m_vir', '1e12 Msol')
        self.__m_vir = units.Unit(self.__m_vir)
        self.__h = kwargs.get('h', 0.7)
        self.__overden = kwargs.get('overden', 200.)
        self.__r_vir = tools.calc_r_vir(self.__m_vir, self.__h, self.__overden)
        self.__r_s = self.__r_vir/self.__pars['c']
        self.__n_particles = int(kwargs.get('n_particles', 1e5))
        self.__logxmax_rho = np.log10(self.__pars['c']) + 2.
        # Make sure to sample well inside the gravitational softening
        self.__logxmin_rho = self.__logxmax_rho - .5*np.log10(self.__n_particles) - 3.
        self.__logxmax_dist_func = np.log10(self.__pars['c']) + 13.
        self.__logxmin_dist_func = self.__logxmax_dist_func - .5*np.log10(self.__n_particles)
        self.__logxmin_dist_func -= 14.
        self.__n_sample_rho = int(kwargs.get('n_sample_rho', 1e4))
        self.__n_sample_dist_func = int(kwargs.get('n_sample_dist_func', 1e2))
        self.__n_sample_dist_func_rho = int(kwargs.get('n_sample_dist_func_rho', 1e4))
        self.__random_seed = kwargs.get('random_seed', 4)
        if 'prng' in kwargs.keys():
            self.__prng = kwargs['prng']
        else:
            self.__prng = np.random.RandomState(self.__random_seed)
        self.__spline_order = kwargs.get('spline_order', 3)
        self.__progress_bar = kwargs.get('progress_bar', False)
        self.__no_bulk_vel = kwargs.get('no_bulk_vel', True)
        self.__x_rho = np.logspace(self.__logxmin_rho, self.__logxmax_rho, self.__n_sample_rho)
        self.__do_velocities = kwargs.get('do_velocities', True)
        self.__gas = kwargs.get('gas', False)
        if 'snap' in kwargs.keys():
            self.sim = kwargs['snap']
        elif self.__gas:
            self.sim = snapshot._new(gas=self.__n_particles)
        else:
            self.sim = snapshot._new(self.__n_particles)

    def __mass(self, x):
        """Calculate enclosed mass in spherical shells"""
        return tools.simpsons_integral(x, lambda x: x*x*self.__profile(x, self.__pars),
            zero=True)

    def __g(self, x):
        """Calculate gravitational acceleration as function of radius"""
        return -self.__mass(x)/x/x

    def __psi(self, x):
        """Calculate effective gravitational potential as funtion of radius"""
        return tools.simpsons_integral(x, self.__g, False, norm_ind=-1)

    def __d2rhodpsi2(self, x):
        """Calculate 2nd derivative of density with respect to potential"""
        ret = self.__profile(x, self.__pars)/self.__g(x)
        ret -= 2.*self.__mass(x)/self.__g(x)/x**3
        ret *= self.__drhodr(x, self.__pars)
        ret += self.__d2rhodr2(x, self.__pars)
        ret /= self.__g(x)**2
        return ret

    def __sample_radii(self):
        """Draws radii from radial mass distribution"""
        masses = self.__mass(self.__x_rho)
        self.__m_max = masses.max()
        no_duplicate = np.where(masses < self.__m_max)
        no_duplicate = np.append(no_duplicate[0], no_duplicate[0][-1] + 1)
        mass = masses[no_duplicate]
        if (mass[:-1] >= mass[1:]).any():
            raise RuntimeError, "Mass array not monotonically increasing"
        self.__x_of_cum_mass_dist_tck = interp.splrep(mass, self.__x_rho[no_duplicate],
            k=self.__spline_order)
        self.__r = interp.splev(self.__prng.uniform(0., self.__m_max, self.__n_particles),
            self.__x_of_cum_mass_dist_tck)

    def __sample_az_angles(self):
        """Draws azimuthal angels from an anisotropic distribution"""
        self.__az = 2.*np.pi*self.__prng.uniform(size=self.__n_particles)

    def __sample_polar_angles(self):
        """Draws polar angels from an anisotropic distribution"""
        self.__polar = np.arccos(1. - 2.*self.__prng.uniform(size=self.__n_particles))

    def __set_positions(self):
        """Converts radii and angles to cartesian postitions"""
        self.__sample_radii()
        self.__sample_az_angles()
        self.__sample_polar_angles()
        self.__pos = np.ones((self.__n_particles, 3))*np.nan
        self.__pos[:,0] = self.__r*np.cos(self.__az)*np.sin(self.__polar)
        self.__pos[:,1] = self.__r*np.sin(self.__az)*np.sin(self.__polar)
        self.__pos[:,2] = self.__r*np.cos(self.__polar)

    def __calc_x_of_psi_tck(self):
        """Calculate interpolation function for inverting potential to radii"""
        no_duplicate = np.where(np.diff(self.__psi_dist_func) < 0)[0]
        no_duplicate = np.append(no_duplicate, no_duplicate[-1] + 1)
        arr = np.ones((len(no_duplicate), 2))
        arr[:,0] = self.__x_dist_func[no_duplicate]
        arr[:,1] = self.__psi_dist_func[no_duplicate]
        arr.view('f8,f8').sort(order='f1', axis=0)
        self.__x_of_psi_tck = interp.splrep(arr[:,1], arr[:,0], k=self.__spline_order)

    def __calc_f(self):
        """Calculates actual distribution function"""
        self.__x_dist_func = np.logspace(self.__logxmin_dist_func, self.__logxmax_dist_func,
            self.__n_sample_dist_func_rho)
        self.__psi_dist_func = self.__psi(self.__x_dist_func)
        self.__calc_x_of_psi_tck()
        self.__d2rhodpsi2_of_x_tck = interp.splrep(self.__x_dist_func,
            self.__d2rhodpsi2(self.__x_dist_func), k=self.__spline_order)

        pos = np.where(self.__psi_dist_func > 0.)
        self.__logpsi_of_logx_tck = interp.splrep(np.log10(self.__x_dist_func[pos]),
            np.log10(self.__psi_dist_func[pos]), k=self.__spline_order)
        # The following 3 lines are directly translated from the non-public Fortran
        # code by Stelios Kazantzidis
        logxs = np.linspace(np.log10(self.__r.min()), np.log10(self.__r.max()),
            self.__n_sample_dist_func)
        self.__e_dist_func = 10**interp.splev(logxs, self.__logpsi_of_logx_tck)
        if (np.diff(self.__e_dist_func) > 0).any():
            raise RuntimeError, "Potential not a monotonic function of radius"
        self.__f_dist_func = np.ones(self.__n_sample_dist_func)*np.nan
        ts = np.pi/2.*(1. - np.linspace(1., 0., self.__n_sample_dist_func_rho)**4)
        dt = np.diff(ts)
        for i, e in enumerate(self.__e_dist_func):
            xs = interp.splev(e*np.sin(ts), self.__x_of_psi_tck)
            integrand = interp.splev(xs, self.__d2rhodpsi2_of_x_tck)*np.sqrt(1. + np.sin(ts))
            integrand = 0.5*(integrand[:-1]+integrand[1:])
            self.__f_dist_func[i] = np.sqrt(e)*(integrand*dt).sum()
        # This normalization is not really necessarry but useful to test results against
        # known analytic solutions
        self.__f_dist_func /= np.sqrt(128.*np.pi**6)

    def __calc_f_of_e_tck(self):
        """Interpolation of distribution function"""
        arr = np.ones((len(self.__e_dist_func), 2))
        arr[:,0] = self.__e_dist_func
        arr[:,1] = self.__f_dist_func
        arr.view('f8,f8').sort(order='f0', axis=0)
        f_of_e_max = arr[:,1].copy()
        # Make sure f_max is monotonically rising
        declining = np.diff(f_of_e_max) < 0.
        while declining.any():
            f_of_e_max[1:][declining] = f_of_e_max[:-1][declining]
            declining = np.diff(f_of_e_max) < 0.
        self.__fmax_of_e_tck = interp.splrep(arr[:,0], f_of_e_max, k=self.__spline_order)
        self.__logf_of_loge_tck = interp.splrep(np.log10(arr[:,0]), np.log10(arr[:,1]),
            k=self.__spline_order)

    def __sample_v(self):
        """Samples velocities (absolute values) from distribution function"""
        self.__psi_of_x_tck = interp.splrep(self.__x_dist_func, self.__psi_dist_func,
            k=self.__spline_order)
        self.__psi_of_r = interp.splev(self.__r, self.__psi_of_x_tck)
        self.__v_escape = np.sqrt(2.*self.__psi_of_r)
        self.__calc_f_of_e_tck()
        self.__f_max = interp.splev(self.__psi_of_r, self.__fmax_of_e_tck)
        rejects = np.ones(self.__n_particles, dtype=int) == 1
        self.__v = np.ones(self.__n_particles)*np.nan
        self.__max_iter = 0
        self.__tot_iter = 0
        for i in range(self.__n_particles):
            if self.__progress_bar:
                if i%10 == 0:
                    sys.stdout.write(' [{0:3d} %'.format(i*100/self.__n_particles))
                    if i*400/self.__n_particles%4 == 0: sys.stdout.write(' | ]')
                    elif i*400/self.__n_particles%4 ==1: sys.stdout.write(' / ]')
                    elif i*400/self.__n_particles%4 ==2: sys.stdout.write(' - ]')
                    elif i*400/self.__n_particles%4 ==3: sys.stdout.write(' \\ ]')
                    sys.stdout.flush()
            j = 0
            f = 1.
            f_rand = 2.
            while(f_rand > f):
                j += 1 # Count iterations
                # The following is equivalent to sampling vx, vy and vz from a unifrom distr.
                self.__v[i] = self.__v_escape[i]*self.__prng.uniform()**(1./3.)
                e = self.__psi_of_r[i] - 0.5*self.__v[i]**2
                f_rand = self.__f_max[i]*self.__prng.uniform()
                f = 10**interp.splev(np.log10(e), self.__logf_of_loge_tck)
            if j > self.__max_iter: self.__max_iter = j
            self.__tot_iter += j
            if self.__progress_bar:
                if i%10 == 0: sys.stdout.write('\b'*11)

    def __set_velocities(self):
        """Convert absolute velocities to cartesian velocities"""
        self.__sample_v()
        self.__v_az = self.__prng.uniform(size=self.__n_particles)*2.*np.pi
        self.__v_polar = np.arccos(1. - 2.*self.__prng.uniform(size=self.__n_particles))
        self.__vel = np.ones((self.__n_particles, 3))*np.nan
        self.__vel[:,0] = self.__v*np.cos(self.__v_az)*np.sin(self.__v_polar)
        self.__vel[:,1] = self.__v*np.sin(self.__v_az)*np.sin(self.__v_polar)
        self.__vel[:,2] = self.__v*np.cos(self.__v_polar)

    def __set_softening(self):
        """Set the gravitational softening as suggested in Power+2003 (2003MNRAS.338...14P)"""
        self.__n_inside_r_vir = (self.__r < self.__pars['c']).sum()
        self.__eps = self.__pars['c']/np.sqrt(self.__n_inside_r_vir)

    def __calc_mc(self):
        """Calculate mass inside scale radius"""
        masses = self.__mass(self.__x_rho)
        self.__mc = interp.splev(self.__pars['c'], interp.splrep(self.__x_rho, masses,
            k=self.__spline_order))

    def __calc_vel_units(self):
        """Calculate conversion factor to convert velocities to km/s"""
        self.__calc_mc()
        self.__vel_fac = (units.G*self.__m_vir/self.__mc/self.__r_s)**(1,2)

    def sample_equilibrium_halo(self):
        """This method actually creates the halo"""
        start = time.clock()
        print('SampleDarkHalo: setting positions ...'),
        sys.stdout.flush()
        self.__set_positions()
        pos_time = time.clock()
        print('done in {0:.2g} s'.format(pos_time-start))
        if self.__do_velocities:
            print('SampleDarkHalo: calculating distribution function ...'),
            sys.stdout.flush()
            self.__calc_f()
            f_time = time.clock()
            print('done in {0:.2g} s'.format(f_time-pos_time))
            print('SampleDarkHalo: setting velocities ...'),
            sys.stdout.flush()
            self.__set_velocities()
            v_time = time.clock()
            print(' done in {0:.2g} s'.format(v_time-f_time) + ' '*10)
        self.__set_softening()
        #self.__calc_r_vir()
        self.sim['mass'] = array.SimArray(np.ones(self.__n_particles)/self.__n_inside_r_vir,
            self.__m_vir)
        #self.sim['mass'].units = self.__m_vir
        self.sim['pos'] = array.SimArray(self.__pos, self.__r_s)
        #self.sim['pos'].units = self.__r_s
        self.sim['eps'] = array.SimArray(np.ones(self.__n_particles)*self.__eps, self.__r_s)
        #self.sim['eps'].units = self.__r_s
        if self.__do_velocities:
            self.__calc_vel_units()
            self.sim['vel'] = array.SimArray(self.__vel, self.__vel_fac)
        else: self.sim['vel'] = np.zeros(self.sim['vel'].shape)
        if self.__no_bulk_vel: self.sim['vel'] -= self.sim.mean_by_mass('vel')
        end = time.clock()
        print('SampleDarkHalo: halo created in {0:.2g} s'.format(end-start))

    def finalize(self):
        self.sim.properties['a'] = 0. # This is necessarry in order to set the time to 0
        self.sim.physical_units(mass='2.325e5 Msol')

class EquilibriumHalo:
    """

    EquilibriumHalo:
    ================

    A class that is initialized with a set of galaxy parameters (e.g. mass, density
    profile, spin parameter etc.) and creates a gas sphere in hydrodynamic equilibrium
    embedded in a dark matter halo that is in dynamic equilibrium. It uses the
    SampleDarkHalo class to sample the dark matter halo and the gas positions.

    """

    def __init__(self, **kwargs):
        self.__kwargs = kwargs
        self.__profile = kwargs.get('profile', density_profiles.alphabetagamma)
        self.__drhodr = kwargs.get('drhodr', density_profiles.dalphabetagammadr)
        self.__d2rhodr2 = kwargs.get('d2rhodr2', density_profiles.d2alphabetagammadr2)
        self.__pars = kwargs.get('pars', {'alpha': 1., 'beta': 3., 'gamma': 1.,
            'c': 10., 'factor': 0.1})
        if self.__profile == density_profiles.alphabetagamma and self.__pars['beta'] <= 3.:
            if 'factor' not in self.__pars.keys(): self.__pars['factor'] = 0.1
        self.__m_vir = kwargs.get('m_vir', '1e12 Msol')
        self.__m_vir = units.Unit(self.__m_vir)
        self.__h = kwargs.get('h', 0.7)
        self.__overden = kwargs.get('overden', 200.)
        self.__r_vir = tools.calc_r_vir(self.__m_vir, self.__h, self.__overden)
        self.__r_s = self.__r_vir/self.__pars['c']
        self.__n_particles = int(kwargs.get('n_particles', 1e5))
        self.__logxmax_rho = np.log10(self.__pars['c']) + 2.
        # Make sure to sample well inside the gravitational softening
        self.__logxmin_rho = self.__logxmax_rho - .5*np.log10(self.__n_particles) - 3.
        self.__logxmin_dist_func = kwargs.get('logxmin_dist_func', -3.)
        self.__logxmax_dist_func = kwargs.get('logxmax_dist_func', 14.)
        self.__n_sample_rho = int(kwargs.get('n_sample_rho', 1e4))
        self.__n_sample_dist_func = int(kwargs.get('n_sample_dist_func', 1e2))
        self.__n_sample_dist_func_rho = int(kwargs.get('n_sample_dist_func_rho', 1e4))
        self.__random_seed = kwargs.get('random_seed', 4)
        if 'prng' in kwargs.keys():
            self.__prng = kwargs['prng']
        else:
            self.__prng = np.random.RandomState(self.__random_seed)
        self.__spline_order = kwargs.get('spline_order', 3)
        self.__progress_bar = kwargs.get('progress_bar', False)
        self.__no_bulk_vel = kwargs.get('no_bulk_vel', True)
        self.__x_rho = np.logspace(self.__logxmin_rho, self.__logxmax_rho, self.__n_sample_rho)
        self.__f_bary = kwargs.get('f_bary', 0.1)
        self.__mu = kwargs.get('mu', 1.3)
        self.__spin_parameter = kwargs.get('spin_parameter', 0.04)
        self.__rot_balanced = kwargs.get('rot_balanced', False)
        # Different gas profiles are not yet implemented or successfully tested
        self.__gas_profile = self.__profile
        self.__gas_pars = self.__pars
        #self.__gas_profile = kwargs.get('gas_profile', density_profiles.alphabetagamma)
        #self.__gas_pars = kwargs.get('gas_pars', {'alpha': 1., 'beta': 3., 'gamma': 1.,
        #   'c': 10., 'factor': 0.1})
        self.__r_s_gas = self.__r_vir/self.__gas_pars['c']
        #self.__vel_prof = kwargs.get('vel_prof', None)
        self.__vel_pars = kwargs.get('vel_pars', {'rs_v': array.SimArray(1., 'kpc'),
            'c': self.__pars['c'], 'prefac': 1., 'factor': 1.})
        self.__n_gas_particles = int(kwargs.get('n_gas_particles', self.__n_particles))
        self.__ang_mom_prof = kwargs.get('ang_mom_prof', am_profiles.bullock_prof)
        self.__ang_mom_pars = kwargs.get('ang_mom_pars', {'mu': self.__mu})
        self.__fname = kwargs.get('fname', 'halo.out')
        # Careful here: as of now, only the output as tipsy files has been successfully tested
        self.__type = {'gadget': gadget.GadgetSnap,
                        'grafic': grafic.GrafICSnap,
                        'nchilada': nchilada.NchiladaSnap,
                        'ramses': ramses.RamsesSnap,
                        'tipsy': tipsy.TipsySnap}[kwargs.get('type', 'tipsy')]
        self.sim = new(dm=self.__n_particles, gas=self.__n_gas_particles)
        self.sim.physical_units()

    def __mass(self, x):
        """Calculate enclosed mass in spherical shells"""
        return tools.simpsons_integral(x, lambda x: x*x*self.__profile(x, self.__pars),
            zero=True)

    def __gas_mass(self, x):
        """Calculate dimensionless enclosed mass in spherical shells"""
        return tools.simpsons_integral(x, lambda x: x*x*self.__gas_profile(x, self.__pars),
            zero=True)#*self.__f_bary ## This is only the dimensionless mass which must not be normalized by f_bary

    def __calc_m_c(self):
        """Calculate dimensionless mass at virial radius"""
        self.__m_c = np.interp(self.__pars['c'], self.__x_rho, self.__mass(self.__x_rho))
        self.__m_c_gas = np.interp(self.__gas_pars['c'], self.__x_rho,
            self.__gas_mass(self.__x_rho))/self.__f_bary

    def __gravity(self, x):
        """Calculate gravitational acceleration as function of spherical radius"""
        return -self.__mass(x)/x/x

    def __make_dark_halo(self):
        """Sample the dark matter component of the halo including equilibrium velocities"""
        self.__kwargs['prng'] = np.random.RandomState(self.__random_seed)
        self.__kwargs['snap'] = self.sim.d
        self._dark_halo = SampleDarkHalo(**self.__kwargs)
        self._dark_halo.sample_equilibrium_halo()
        #self._dark_halo.finalize()
        self.sim.d['mass'] *= (1. - self.__f_bary)

    def __make_gas_sphere(self):
        """Sample the gas component of the halo but set velocities to 0"""
        prng = np.random.RandomState(self.__random_seed+1)
        self._gas = SampleDarkHalo(profile=self.__gas_profile, pars=self.__gas_pars,
            m_vir=self.__m_vir, h=self.__h, overden=self.__overden,
            logxmin_rho=self.__logxmin_rho, logxmax_rho=self.__logxmax_rho,
            n_samlple_rho=self.__n_sample_rho, n_particles=self.__n_gas_particles,
            prng=prng, do_velocities=False, snap=self.sim.g)
        self._gas.sample_equilibrium_halo()
        #self._gas.finalize()
        self.sim.g['mass'] *= self.__f_bary
        self.sim.g['pressure'] = array.SimArray(np.zeros(self.__n_gas_particles),
            'g cm**-1 s**-2')

    def __calc_enclosed_mass_of_R(self, R):
        """Calculate enclosed mass as function of polar distance"""
        z = np.append(0, np.logspace(-4, 0, 1000))
        if not 0 in R:
            R = array.SimArray(np.append(0, R), R.units)
        Rs, Zs = np.meshgrid(R, z)
        Rs = array.SimArray(Rs, R.units)
        c = self.__gas_pars['c']
        rs = self.__r_vir/c
        integrand = self.__gas_profile(np.sqrt(Rs*Rs+(c*c*rs*rs-Rs*Rs)*Zs*Zs).in_units(R.units),
            self.__gas_pars)*Rs
        integrand[np.where((Rs==0)*(Zs==0))] = 0.
        integrand = 0.5*(integrand[1:,]+integrand[:-1,])
        dz = z[1:] - z[:-1]
        rintegrand = (integrand.T*dz).T.sum(axis=0)
        zz = c*c*rs*rs-R*R
        zz[np.where(zz<0)] *= 0
        rintegrand *= np.sqrt(zz)
        rintegrand = 0.5*(rintegrand[1:]+rintegrand[:-1])
        dR = R[1:] - R[:-1]
        return 4.*np.pi*((rintegrand*dR).cumsum())*self.__f_bary
        #self.__enclosed_gas_mass_tck = interp.splrep(R[1:], enclosed_gas_mass)

    def __invert_ang_mom_profile(self):
        """
        Calculate an interpolating function that inverts the angular momentum profile,
        i.e. it calculates j as a function of the fraction of virial mass with less
        angular momentum
        """
        j = np.linspace(0, 1, 1000)
        m = self.__ang_mom_prof(j, self.__ang_mom_pars)
        self.__inverse_ang_mom_prof_tck = interp.splrep(m, j, k=1)

    def __interpolate_jz_of_R(self):
        """Calculate angular momentum as a function of axisymmetric distance"""
        R = array.SimArray(np.logspace(np.log10(self._gas.sim['rxy'].min()),
            #np.log10(self._gas.sim['rxy'].max()), 1000), self.__r_s)
            np.log10(self._gas.sim['rxy'].max()), 1000),
            self._gas.sim['rxy'].units).in_units(self.__r_s)
        m_encl = self.__calc_enclosed_mass_of_R(R)
        m_encl /= m_encl.max()
        self.__invert_ang_mom_profile()
        jz = interp.splev(m_encl, self.__inverse_ang_mom_prof_tck)
        self.__jz_of_R_tck = interp.splrep(R.in_units('kpc'), jz, k=1)

    def __calc_j_max(self):
        """Calculate maximum angular momentum for Bullock profiles"""
        self.__j_max = ((units.G*self.__m_vir*self.__r_vir)**(1,2)) * self.__spin_parameter
        self.__j_max *= np.sqrt(2.) / (1.-self.__mu) / (self.__mu*np.log(1.-1./self.__mu)+1.)

    def __set_gas_velocities(self):
        """Set the gas velocities according to the specified angular momentum profile"""
        self.__interpolate_jz_of_R()
        vc = array.SimArray(interp.splev(self._gas.sim['rxy'].in_units('kpc'),
            self.__jz_of_R_tck), self.__j_max)
        vc /= self._gas.sim['rxy']
        # Truncate velocity profile smoothly outside R_vir
        r_v = self.__r_vir
        vc *= tools.outer_smooth_cutoff(self._gas.sim['r'].in_units(r_v).view(np.ndarray),
            self.__vel_pars['factor'])
        # Quick and dirty hack due to some weird behaviour of the 'az' derived array in pynbody
        az = np.arctan2(self._gas.sim['y'], self._gas.sim['x'])
        #self._gas.sim['vel'].units = units.Unit('km s**-1')
        self._gas.sim['vel'][:,:-1] = (np.array([-np.sin(az),
            np.cos(az)])*vc.in_units(self.sim['vel'].units)).transpose()
        self._gas.sim['vel'][:,-1] = np.zeros(self.__n_gas_particles)

    def __calc_gravity_pressure(self):
        """
        Solve the equation of hydrostatic equilibrium neglecting centrifugal effects of rotation
        """
        integrand = lambda x: self.__gravity(x)*self.__gas_profile(x*self.__gas_pars['c']/
            self.__pars['c'], self.__gas_pars)
        self.__p = array.SimArray(tools.simpsons_integral(self.__x_rho, integrand, norm_ind=-1))
        self.__calc_m_c()
        unit = 4.*np.pi*self.__pars['c']*self.__gas_pars['c']**3*self.__overden**2
        unit /= 9.*self.__m_c*self.__m_c_gas
        unit *= units.G*self.__r_vir**2*tools.calc_rho_crit(self.__h)**2
        self.__p.units = tools.sim_array_to_unit(unit)
        self.sim.g['pressure'] += np.interp(self.sim.g['r'].in_units(self.__r_s),
            self.__x_rho, self.__p.in_units(self.sim.g['pressure'].units))

    def __calc_temp(self):
        """Calculate the gas temperature based on its density and pressure"""
        # units
        rho_0 = 2./3.*self.__gas_pars['c']**3*self.__overden/self.__m_c_gas
        rho_0 *= tools.calc_rho_crit(self.__h)
        # density 1st guess
        dens = array.SimArray(np.interp(self._gas.sim['r'].in_units(self.__r_s_gas),
            self.__x_rho, self.__gas_profile(self.__x_rho, self.__gas_pars)), rho_0/2.)
        temp = (self.sim.g['pressure']/dens/units.k*units.m_p).in_units('K')
        self.sim.g['temp'] = temp
        tools.iterate_temp(self._gas.sim)

    def make_halo(self):
        """
        Put everything together

        This method actually creates the halo
        """
        self.__make_dark_halo()
        self.__make_gas_sphere()
        self.__calc_j_max()
        self.__invert_ang_mom_profile()
        self.__set_gas_velocities()
        self.__calc_gravity_pressure()
        self.__calc_temp()

    def finalize(self):
        """
        Prepare IC snapshot and write it to disc
        """
        self.sim.properties['a'] = 0.
        self.sim.properties['time'] = 0.
        self.sim.g['metals'] = 0.
        del(self.sim['pressure'], self.sim['mu'])
        if self.__type == tipsy.TipsySnap:
            self.sim.physical_units(mass='2.325e5 Msol')
        else:
            self.sim.physical_units()
        self.sim.write(self.__type, self.__fname)
