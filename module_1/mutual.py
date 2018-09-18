from sklearn import metrics as mtr
import pandas as pd
import numpy as np
import operator

from sklearn import preprocessing

class mutual_information(object):

    def __init__(self, data, target):
        self.X = []
        self.data = data
        self.target = target

    @staticmethod
    def feature_descritization(variables):

        def remove_errors(variables):
            #variables.drop_duplicates(keep='first', inplace=False)
            
            # for i in range(data.shape[0]):
            #     data[i] = np.array(list(filter(lambda v: v == v, data[i]))).astype(np.float)
            return variables

        cl_data = variables
        cl_data = remove_errors(cl_data)
        cl_data = np.array(cl_data)

        X_tezheng = cl_data[:, 0:cl_data.shape[1]]
        X_ls = np.empty((X_tezheng.shape[0], X_tezheng.shape[1]), dtype=int)
        for i in range(X_ls.shape[1]):
            k = 2
            X_ls[:, i] = pd.cut(X_tezheng[:, i], k, labels=range(k))
        return X_ls, X_tezheng, cl_data

    def shuffle_data(self):
        # disorder the new feature by lines
        permutation_way = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation_way, :]
        self.Y = self.Y[permutation_way]

    def compute_mutual(self):

        data=np.column_stack([self.data, self.target])

        X_ls, mx_data, cl_data = self.feature_descritization(data)

        y = self.target

        hu_results = []
        for i in range(X_ls.shape[1]):
            x_single = X_ls[:,i]

            hu_result = mtr.mutual_info_score(y,x_single)
            hu_results.append(hu_result)

        list1 = list(range(data.shape[1]))
        middle1 = dict(zip(list1, hu_results))
        middle2 = dict(sorted(middle1.items(), key=operator.itemgetter(1)))

        feature = list(middle2.keys())
        results = []
        for i in range(2):
            results.append(feature[-(1 + i)])

        x_data = np.zeros((mx_data.shape[0], 8))

        j = 0
        for i in results:
            x_data[:, j] = mx_data[:, i]
            j += 1

        return x_data,y, feature




