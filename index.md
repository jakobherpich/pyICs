# pyICs documentation
Here you can find the documentation of my [pyICs](https://github.com/jakobherpich/pyICs) code. If you have further questions regarding the code, you can either submit an [issue](https://github.com/jakobherpich/pyICs/issues) or drop me an [email](mailto:herpich@mpia.de).

## What is pyICs?
[pyICs](https://github.com/jakobherpich/pyICs) is a python package to create initial condition (IC) files for N-body simulations of the formation of isolated galaxies. It uses the [pynbody](https://github.com/pynbody/pynbody) analysis package to create the actual IC files. pyICs generates dark matter halos (DM) in dynamical equilibrium which host a rotating gas sphere. The DM particle velocities are drawn from the equilibrium distribution function (Kazantzidis et al. 2004 ApJ, 601, 37). The gas sphere has a Bullock et al. (2001 ApJ, 555, 240) angular momentum profile. The DM and the gas share the same 3D radial density profile. The code natively supports the αβγ-models: ρ ~ (r/a)<sup>-γ</sup>[1+(r/a)<sup>α</sup>]<sup>(γ-β)/α</sup>. If γ <= 3, the profiles are smoothly truncated outside the virial radius.

The radial profile can be arbitrary as long as python functions for the profile itself and its first and second derivative with radius are given.

## Installation
If you have pyhton configured with distutils the following should get you started:
```
$ git clone https://github.com/jakobherpich/pyICs.git
$ cd pyICs
$ python setup.py install
$ cd ..
$ python
>>> import pynbody
```

## Using the code
The code will create an initial conditions file that can be evolved with a Hydrodynamics solver. The code has only been tested for the *tipsy* file format which is supported by [Gasoline](http://adsabs.harvard.edu/abs/2004NewA....9..137W) and [ChaNGa](https://github.com/N-BodyShop/changa). It should, however, also support Grafic IC Snaps, Gadget, Ramses and N-Chilada snapshots. Compatibility with these formats has not been tested. I would really appreciate any effort to test these other formats. If you do so please inform me about your experience and possible hacks to make it work.

### Tipsy units
In the default case of tipsy file outputs the length and mass units are:
```
dKpcUnit = 1.
dMsolUnit = 2.325e5
```
Thus, the two line above need to be included in the `param` file.

### The simple way
Use the alpha-beta-gamma models. In this case the only thing you need to do, is to specify the the values for α, β and γ:
```
>>> from pyICs import create_ics
>>> pars = {'alpha': 1., 'beta': 3., 'gamma': 1., 'c': 10.}
>>> myhalo = creat_ics(pars=pars)
```
This piece of code will create a 10<sup>12</sup> Msol halo with an NFW density profile and a spin parameter of 0.04. The initial conditions file will be saved in the working directory as `halo.out`. The default parameters can be changed (see [Parameters](#parameters)).

### The versatile way
You can also define the density profile yourself. In this case you have to specify the corresponding first and second derivative as these are needed to compute the distribution function of the stable halo. Note that the profiles do not need to be properly normalized as this will be done by the code according to a specified virial mass of the halo (The default value is 10<sup>12</sup> Msol with Hubble constant h=0.7 and mean overdensity &lt;rho&gt;/rho_crit(h)=200.). All three of these functions must accept two arguments: radius (array like), pars (python dictionary with possible parameters, can be empty `{}`)
```
import numpy as np
>>> from pyICs.convenience import create_ics
>>> def rho(r, pars):
    sig = pars['sig']
    return np.exp(-0.5*r**2/sig**2)
>>> def drhodr(r, pars):
    sig = pars['sig']
    return -r/sig**2*rho(r, pars)
>>> def d2rhodr2(r, pars):
    sig = pars['sig']
    return -1./sig**2*rho(r, pars)-r/sig**2*drhodr(r, pars)
>>> pars = {'sig': 10}
>>> sim = create_ics(profile=rho, drhodr=drhodr, d2rhodr2=d2rhodr2, pars=pars, length_unit='1 kpc')
```
Again the resulting initial conditions file will be saved as `halo.out`.

## Parameters
Here we present a list of the most important parameters to be passed to the code in `key=value` style:

Key | Default | Description 
:-----|:---------|:-------------
`m_vir`| `'1e12 Msol'` | The virial mass of the halo. Can be string with units or a float (then units are Msol)
`profile`| αβγ | A python function for the density profile as a function of radius, can have arbitrary normalization and must accept two arguments: radius and a python dictionary with function parameters (key `'c'` is reserved for halo concentration of αβγ-models)
`drhodr`| dαβγ/dr | The derivative of `profile`, must have same normalization as `profile` and use the same parameter dictionary 
`d2rhodr2` | d<sup>2</sup>αβγ/dr<sup>2</sup> |The 2nd derivative of `profile`, must have same normalization as `profile` and use the same parameter dictionary 
`pars`| NFW with exponential cutoff | Python dictionary that is passed as 2<sup>nd</sup> argument to `profile`, `drhodr` and `d2rhodr2` 
`n_particles`|`1e5`| Number of gas and DM particles respectively 
`n_gas_particles`|`n_particles`| Number of gas particles (overwrites value set by `n_particles`, implementation not yet thoroughly tested)
`f_bary`| `0.1` | Gas mass fraction 
`mu`| `1.3` | Shape parameter for Bullock+2001 angular momentum profile 
`spin_parameter`|`0.04`|Spin parameter initial halo
`f_name`|`'halo.out'`|Filename of initial conditions file

## Acknowledging pyICs
If you use pyICs for your work scientific work please mention it along with my name in the acknowledgments:
*This work made use of the open-source python initial condition creation package {\sc pyICs} written by Jakob Herpich (\url{https://github.com/jakobherpich/pyICs).*

Additionally you can cite my paper (http://adsabs.harvard.edu/abs/2015arXiv151104442H) which is part of a series of papers in which pyICs was first used.
