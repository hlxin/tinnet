#!/usr/bin/env python

import os
import warnings

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    msg = ("Please install numpy (version 1.7.0 or greater) before installing "
           "Piml. (Piml uses numpy's installer so it can compile the fortran "
           "modules with f2py.) You should be able to do this with a command"
           " like:"
           "   $ pip install numpy")
    raise RuntimeError(msg)


name = 'piml'
version = open(os.path.join('piml', 'VERSION')).read().strip()
description = 'Physics-Informed Machine Learning'
long_description = open('README').read()
packages = ['piml', 'piml.feature', 'piml.regression',
            'piml.model', 'piml.stats']
Package_dir = {'piml': 'piml', 'feature': 'feature',
               'regression': 'regression', 'model': 'model', 'stats': 'stats'}
classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2.6',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.3']
install_requires = ['numpy>=1.7.0', 'matplotlib', 'ase', 'pyzmq',
                    'pexpect']
ext_modules = [Extension(name='piml.fmodules',
                         sources=['piml/model/neuralnetwork.f90',
                                  'piml/feature/gaussian.f90',
                                  'piml/feature/cutoffs.f90',
                                  'piml/feature/zernike.f90',
                                  'piml/model.f90'])]
author = 'Hongliang Xin'
author_email = 'hxin@vt.edu'
url = 'https://github.com/hlxin/piml'
package_data = {'piml': ['VERSION']}

scripts = ['tools/piml-compress', 'tools/piml-plotconvergence']

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
    if 'piml.fmodules' in ex.args[0]:
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
    warnings.warn('Installed Piml without fortran modules since no fortran '
                  'compiler was found. The code may run slow as a result.')
