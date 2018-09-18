'''
this section used for testing all framework
'''

from module_1 import turnament_selection, min_max_scaler,\
mutual_information, micro_data, dataset_types
from module_1.turnament_selection import evol_selection

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import preprocessing, linear_model
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import feature_selection

import warnings
warnings.filterwarnings("ignore")

import cProfile, tempfile, os, pstats

def profile(column='time', list=5):
    def _profile(function):
        def __profile(*args, **kw):
            s = tempfile.mktemp()
            profiler = cProfile.Profile()
            profiler.runcall(function, *args, **kw)
            profiler.dump_stats(s)
            p = pstats.Stats(s)
            p.sort_stats(column).print_stats(list)
        return __profile
    return _profile


import time
def time_testing(function):
    global start_time, time
    start_time = time.time()
    function
    print("---lasted %s seconds ---" % str(time.time() - start_time))


@profile('time', 6)
def test_run():
    print('check functions errors')
    '''
        run testing functions
    '''

    print('run test: microdata')
    d = micro_data()

    print('run test: dataset_types')
    dataset_types(d)

    print('run test: min_max_scaler data')
    min_max_scaler(np.array(d)[:, 1:10])

    print('run test: mutual_information')
    mt = mutual_information(data=np.array(d)[:, 1:10],target=np.array(d)[:, 10])
    mt.compute_mutual()

@profile('time', 6)
def run_test_models():

    print('run test: gene_selection')

    '''
        testing genetic algorithm with logistic regression
    '''

    d = micro_data()

    X = np.array(d)[:, 1:10]
    Y = np.array(d)[:, 10]

    Y = Y.astype('int')

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    gen = evol_selection(n_generation=30, model=model)
    gen.evol_gene(X,Y)

run_test_models()