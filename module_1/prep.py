import pandas as pd
import numpy as np

def check_values_tupe(X):
    if (isinstance(X, float)):
        print("var is float.")
    elif (isinstance(X, int)):
        print("var is integer.")

def dataset_types(data):
    if data.__class__.__name__ == 'ndarray':
        print('dataset has type: \n', data.__class__.__name__)
    else:
        print('dataset type is \n:',data.__class__.__name__)
        print('and changed to \n: ',data.__class__.__name__)

# normalized processing
def min_max_scaler(X):
    for i in range(X.shape[1]):
        Xmax = np.max(X[:, i])
        Xmin = np.min(X[:, i])
        for j in range(X.shape[0]):
            if (Xmax - Xmin) == 0:
                X[j][i] = 1
            else:
                X[j][i] = (X[j][i] - Xmin) / (Xmax - Xmin)
    return X




