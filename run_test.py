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

def run_test_models_b():
    '''
        testing genetic algorithm with logistic model
    '''

    print('run test: with mutual_information')
    d = micro_data()
    mt = mutual_information(np.array(d)[:, 1:10],np.array(d)[:, 10])
    X, Y, features = mt.compute_mutual()

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    gen = evol_selection(n_generation=30, model=model)
    gen.evol_gene(X,np.array(Y).astype('int'))


def banchmark_evolution_algorithm():
    d = micro_data()

    X = np.array(d)[:, 1:10]
    Y = np.array(d)[:, 10]
    model = linear_model.LogisticRegression()

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    gen = evol_selection(n_generation=30, model=model)
    gen.evol_gene(X,Y.astype('int'))
    chose_features= gen.emp_list

    scores=cross_val_score(model, X[:,(chose_features)],Y.astype('int'), scoring='accuracy', cv=5).mean()
    print("EVOLUTION_GE features Accuracy: {}".format(scores))

    scores = cross_val_score(model, X[:, (chose_features)], Y.astype('int'), scoring='roc_auc', cv=5).mean()
    print("EVOLUTION_GE features roc_acu: {}".format(scores))

banchmark_evolution_algorithm()

def banckmark_feature_selection():
    d = micro_data()

    X = np.array(d)[:, 1:10]
    Y = np.array(d)[:, 10]

    class PipelineRFE(Pipeline):
        def fit(self, X, y=None, **fit_params):
            super(PipelineRFE, self).fit(X, y, **fit_params)
            self.coef_ = self.steps[-1][-1].coef_
            return self

    pipe = PipelineRFE(
        [
            ('std_scaler', preprocessing.StandardScaler()),
            ("LR", linear_model.LogisticRegression(random_state=42))
        ]
    )
    _ = StratifiedKFold(random_state=42)
    print("Scores for validation banchmark sklearn RFE")
    feature_selector_cv = feature_selection.RFECV(pipe, cv=5, step=2, scoring="accuracy")\
    .fit(X, Y.astype('int'))

    print(pipe.__class__.__name__+" accuracy is {}".format(feature_selector_cv.grid_scores_.mean()))

    feature_selector_cv = feature_selection.RFECV(pipe, cv=5, step=2, scoring="log_loss")\
    .fit(X, Y.astype('int'))

    print(pipe.__class__.__name__+" log_loss is {}".format(feature_selector_cv.grid_scores_.mean()))

    feature_selector_cv = feature_selection.RFECV(pipe, cv=5, step=2, scoring="roc_auc")\
    .fit(X, Y.astype('int'))

    print(pipe.__class__.__name__+" auc is {}".format(feature_selector_cv.grid_scores_.mean()))
