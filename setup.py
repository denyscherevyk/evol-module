import os

from setuptools import setup, find_packages

README = os.path.join(os.path.dirname(__file__), 'readme.rst')
long_description = open(README).read() + '\n\n'


Requires=[
            'numpy',
            'scipy',
            'pandas',
            'tables',
            'scikit-learn',
            'nose>=1.0'
]

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Software Development :: Libraries',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
]

metadata = dict(
    name='evol_module',
    version='0.1.0',
    description=("Genetic Algorithm Package OC"),
    url="https://github.com/denyscherevyk/evol_module",
    long_description=long_description,
    license="GNU General Public License v3",
    classifiers=classifiers,
    install_requires=Requires,
    test_suite='nose.collector',
    test_requests=['Nose'],
    scripts=[],
    package_data={'eqmagene': ['LICENSE.txt', 'readme.rst', 'release.rst']},
    zip_safe=False,
    namespace_packages=[])

setup(**metadata)

'''

python setup.py --help-commands
python setup.py sdist // It creates a release tree where everything needed to run the package is copied
python setup.py bdist // create a binary distribution 
python setup.py install --record installation.txt // command installs the package into Python. 
python setup.py develop
python setup.py register

del ./.git/index.lock

# git control
git init .
git config --global user.email "denys.cherevyk@gmail.com"
git config --global user.name "denyscherevyk"
git config --get-all "denyschrevyk"
git remote add origin https://github.com/denyscherevyk/evol-module.git
git add .
git commit -m "Initial commit"
git pull origin master
'''
