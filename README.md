#pyICs

[pyICs](https://github.com/jakobherpich/pyICs) is a software for creating initial conditions (ICs) to simulate the formation of isolated galaxies. It was designed to create IC files in tipsy format (PKDGRAV/Gasoline/ChaNGa, successfully tested) but should also work for Gadget/Ramses/nchilada (all not tested) files.

[pyICs](https://github.com/jakobherpich/pyICs) depends heavily on the [pynbody](https://github.com/pynbody/pynbody) package as it uses pynbody to create the actual IC files.

##Getting started

If you have pyhton configured with distutils the following should get you started:
```
$ git clone https://github.com/jakobherpich/pyICs.git
$ cd pyICs
$ python setup.py install
$ cd ..
$ python
>>> import pynbody
```
