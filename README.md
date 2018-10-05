
Genetic Selection Features Framework
=======

Genetic algorithm (GA) is a process of natural selection that belongs to the larger class of evolutionary algorithms (EA).
Commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operators such as mutation, crossover and selection.

An optimizing module prepared in the Python programing language, additionally a partially supported in C environment.

Steps run algorithm
=======
1. preprocessing data: remove NaN counts, remove duplicate rows, normalized dataset
2. features descritization
3. calculate mutual information
4. disorders new features by lines
5. diving dataset
6. balanced procesing
7. define type of model
8. setting the hyper-parameters: population size=80, generation length=30, mutate probability=.2, crossover probability=1, kfold=5
9. run start Genetic Algorithm Selection

Flowchart
=======
[1]. Tournament selection
a. set parameters: M, N, pc, pm, k
b. set population N
c. set random strings of k
c.a. evaluate and chose best one
c.b. repat c. once
d. set 2 parent
f. start crossovering at pc
e. set 2 children
g. start mutation children 2 at pm
g.a.  when n = N/2 then m = M
else n = n + 1 end
g.b. repeat c. to finish all generations


Install
=======

Install with pip.::

    $ pip install eqmagene

To install package from github. ::

    $ pip install git+git://github.com/jkbr/httpie.git

or::

    $ git clone https://github.com/jkbr/httpie.git

After just run the setup.py file from install directory. ::

    $ python setup.py install


How run framework
========

You can run the package on your own data, important must bleed out that this data is like as array.
::

    >>> from eqmagene import gene_selection
    >>> X = np.array(d)[:, 1:10]
    >>> Y = np.array(d)[:, 10]
    >>> Y = Y.astype('int')
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> gen = gene_selection(gen=30, model=model)
    >>> gen.evol_gene(X,Y)

Let's use the Genetoc Selection compare with Logistic Regression to run 5-fold cross
validation on the training data.
::

    >>> [FIT RESULT GEN# 1], AV: 0.4450, MIN: 0.4450, SD: 0.0000, MAX: 0.4450
    >>> [FIT RESULT GEN# 2], AV: 0.3950, MIN: 0.3450, SD: 0.0500, MAX: 0.4450
    >>> [FIT RESULT GEN# 3], AV: 0.3475, MIN: 0.3450, SD: 0.0025, MAX: 0.3500
    ........................................................................
    ........................................................................
    >>> [FIT RESULT GEN# 29], AV: 0.3000, MIN: 0.3000, SD: 0.0000, MAX: 0.3000
    >>> [FIT RESULT GEN# 30], AV: 0.3000, MIN: 0.3000, SD: 0.0000, MAX: 0.3000
    >>>
    >>> Final Solution [[0. 0. 0. 1. 0. 0. 0. 1. 0.]]
    >>> Highest Accuracy [0.7]



License
=======

Eqmagene is distributed under the the GPL v3+. See LICENSE file for details.
Where indicated by code comments parts of NumPy, Pandas, Sklearn are included in Eqmagene. The
they license appears in the licenses directory.
