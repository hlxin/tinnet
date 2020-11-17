#!/usr/bin/env python

import os
import warnings

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    msg = ("Please install numpy (version 1.7.0 or greater) before installing "
           "TinNet. (TinNet uses numpy's installer so it can compile the fortran "
           "modules with f2py.) You should be able to do this with a command"
           " like:"
           "   $ pip install numpy")
    raise RuntimeError(msg)


name = 'tinnet'
version = open(os.path.join('tinnet', 'VERSION')).read().strip()
description = 'Physics-Informed Machine Learning'
long_description = open('README').read()
packages = ['tinnet', 'tinnet.feature', 'tinnet.regression',
            'tinnet.model', 'tinnet.stats']
package_dir = {'tinnet': 'tinnet', 'feature': 'feature',
               'regression': 'regression', 'model': 'model', 'stats': 'stats'}
classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2.6',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.3']
install_requires = ['numpy>=1.7.0', 'matplotlib', 'ase', 'pyzmq',
                    'pexpect']
ext_modules = [Extension(name='tinnet.fmodules',
                         sources=['tinnet/model/neuralnetwork.f90',
                                  'tinnet/feature/gaussian.f90',
                                  'tinnet/feature/cutoffs.f90',
                                  'tinnet/feature/zernike.f90',
                                  'tinnet/model.f90'])]
author = 'Hongliang Xin'
author_email = 'hxin@vt.edu'
url = 'https://github.com/hlxin/tinnet'
package_data = {'tinnet': ['VERSION']}

scripts = ['tools/tinnet-compress', 'tools/tinnet-plotconvergence']

try:
    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          packages=packages,
          package_dir=package_dir,
          classifiers=classifiers,
          install_requires=install_requires,
          scripts=scripts,
          ext_modules=ext_modules,
          author=author,
          author_email=author_email,
          url=url,
          package_data=package_data,
          )
except SystemExit as ex:
    if 'tinnet.fmodules' in ex.args[0]:
        warnings.warn('It looks like no fortran compiler is present. Retrying '
                      'installation without fortran modules.')
    else:
        raise ex
    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          packages=packages,
          package_dir=package_dir,
          classifiers=classifiers,
          install_requires=install_requires,
          scripts=scripts,
          ext_modules=[],
          author=author,
          author_email=author_email,
          url=url,
          package_data=package_data,
          )
    warnings.warn('Installed TinNet without fortran modules since no fortran '
                  'compiler was found. The code may run slow as a result.')
