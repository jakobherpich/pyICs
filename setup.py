#from __future__ import print_function
from setuptools import setup#, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import pyICs

here = os.path.abspath(os.path.dirname(__file__))

#def read(*filenames, **kwargs):
#    encoding = kwargs.get('encoding', 'utf-8')
#    sep = kwargs.get('sep', '\n')
#    buf = []
#    for filename in filenames:
#        with io.open(filename, encoding=encoding) as f:
#            buf.append(f.read())
#    return sep.join(buf)
#
#long_description = read('README.txt', 'CHANGES.txt')

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='pyICs',
    version='0.10',
    url='http://github.com/jakobherpich/pyICs/',
    license='GNU General Public License v3 or later (GPLv3+)',
    author='Jakob Herpich',
    tests_require=['pytest'],
    install_requires='pynbody',
    cmdclass={'test': PyTest},
    author_email='herpich@mpia.de',
    description='set up initial conditions for simulations of isolated galaxies',
    #long_description=long_description,
    packages=['pyICs'],
    include_package_data=True,
    platforms='any',
    test_suite='pyICs.test.test_pyICs',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
    extras_require={
        'testing': ['pytest'],
    }
)
