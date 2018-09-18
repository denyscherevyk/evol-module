"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2017
"""

from sklearn.model_selection import StratifiedKFold

from module_1.mutual import mutual_information
from random import randint

import random as rd
import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

from concurrent import futures

class BaseMeta(type):
    def __new__(cls, name, based, body):
        if name != 'mutual_information' and not'variable' in body:
            raise TypeError('bad used class')
        return super().__new__(cls, name, based, body)



class fitness_function():
    def __call__(self, mut_chroms, X, Y, k=5, model=None, shuffle=False):
        """

        Class for computing fitness of the given chromosome.

        Parameters
        ---------

        mut_chroms: array_like
                 Mutated chromosome per population.

        X: array_like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                    and n_features is the number of features.

        Y: array_like, shape = [n_samples]
                     Target Values.

        k: int, optional
                Set numbers of kfold.

        model: array_like
                    Example models to generate fitness function

        shuffle: bool, optional
                    Set random shuffle data in stratified kfold.

        """

        if k < 2 or mut_chroms is None or X is None or Y is None:
            raise ValueError("GE: fitness function setting wrong values ")

        elif  model is None:
            raise NotImplementedError("GA: fitness function must be implemented subclasses")

        try:

            result = []
            result = list(filter(lambda x: mut_chroms[x] == 1,
                                   range(0, len(mut_chroms))))

            X = X[:, result]

            skf = StratifiedKFold(n_splits=k, shuffle=shuffle)

            P = 0
            for index,(train_index, test_index) in enumerate(skf.split(X, Y)):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                model1 = model.fit(X_train, Y_train)
                acurracy = model1.score(X_test, Y_test)
                fitness = 1 - acurracy
                P += fitness
            return (P / k)

        except ValueError("arrays is not of the right shape") as err:
            raise log.error(err)

        else:
            pass

        finally:
            pass




class evol_selection(object):
    def __init__(self, p_mutation=.2, p_covering=.95, n_population=80, n_generation=30, kfold=5, model=None):

        """
            This class realizes GA over features in data,
            tries to find a combination of the features in data with the best fitness value
            for descrimination significant Y.

            Parameters
            -----------
            model: class_like
                    Now the algorithm suported onli scikit-learn framefork model but posiblbe try others,

            kfold: int
                    Number of splits for cross_validation to calculate fitness.

            n_population: int
                    population size, can calculated N =  1 - log(2(1/1-p1(1/l))
                    where l - locus length in chromosome. P - p_crosover.

            p_covering: float
                    Probability of crossover. Recommended values are 80-95%. Default is 95% (0.95).

            p_mutation:  {float}
                    Probability of mutation. Recommended values are 0.5-1%. Default is 0.5% (0.05).

            n_generation: int
                    Maximum number of GA generations.


        """

        self.p_mutation = p_mutation; self.p_covering = p_covering
        self.n_population = n_population; self.n_generation = n_generation
        self.kfold=kfold; self.model=model; self.emp_list=[]

        self.fitness = fitness_function()
        self._parameters_first_debug()


    def _parameters_first_debug(self):
        """
            Method verifies common input parameters of a genetic algorithm.
        """
        if self.p_mutation >1 or self.p_covering >1 or\
            self.kfold<2 or self.n_population is None or\
            self.n_generation is None:
            raise ValueError("GA: setting wrong values")


    @staticmethod
    def _mutation(child, p_mutation=.2):
        """
        This method mutates (inverses bits) the given chromosome.

        Parameters
        -----
            child: array_like
                childs chromosome values for mutation.

            p_mutation: float, optional
                 mutation probablity.

        """

        result = []

        t = 0
        while t < len(child):
            ran_mut_1 = np.random.rand()
            if ran_mut_1 < p_mutation:

                if child[t] == 0:
                    child[t] = 1
                else:
                    child[t] = 0

                t = t + 1
                result = child
                return result
            else:
                result = child
                return result


    @staticmethod
    def __preparing_data(X, Y):
        """
            This method are prepared data to computing GA.
        """
        mut = mutual_information(data=X, target=Y)
        X, Y, features = mut.compute_mutual()
        return X,Y,features


    def __random_chromosoms(self):
        """
            This method generated chromosome
        """
        if self.X is not 0:
            self.chromosome_len = self.X.shape[1]
        else:
            raise ValueError("GA: __random_chromosoms(): wrong values")

        randlist = lambda n: [randint(0, 1) for b in range(1, n + 1)]
        self._chromosoms = randlist(self.chromosome_len)
        return self._chromosoms

    @staticmethod
    def __optim_count(all_ingeneration_x, maximize=False):
        R = []
        t=0
        if maximize == False:
            for i in all_ingeneration_x:
                if (all_ingeneration_x[t, :1]) <= min(all_ingeneration_x[:, :1]):
                    R = all_ingeneration_x[t, :]
                t = t + 1
            return R

        else:
            for i in all_ingeneration_x:
                if (all_ingeneration_x[t, :1]) >= max(
                        all_ingeneration_x[:, :1]):
                    R = all_ingeneration_x[t, :1]
                t = t + 1
            return R

    def evol_gene(self,X,Y):
        """
            X:  array_like, shape = [n_samples, n_features]
                     Training vectors, where n_samples is the number of samples
                     and n_features is the number of features.

            Y: array_like, shape = [n_samples]
                     Target Values
        """

        self.X, self.Y = X, Y

        if self.X is None or self.Y is None:
            raise  NotImplementedError('GA: Mehthod must be implemented subclasses X and Y')


        self.chromosoms = self.__random_chromosoms()

        # create initial population

        n_list = np.empty((0, len(self.chromosoms)))
        i=0
        while i < (self.n_population):
            rd.shuffle(self.chromosoms)
            n_list = np.vstack((n_list, self.chromosoms))
            i+=1


        results_generation_X = []
        results_w_generation_X = []

        one_final_guy = np.empty((0, len(self.chromosoms) + 2))
        one_final_guy_final = []

        min_for_all_generations_for_mut_1 = np.empty((0, len(self.chromosoms) + 1))
        min_for_all_generations_for_mut_2 = np.empty((0, len(self.chromosoms) + 1))

        min_for_all_generations_for_mut_1_1 = np.empty((0, len(self.chromosoms) + 2))
        min_for_all_generations_for_mut_2_2 = np.empty((0, len(self.chromosoms) + 2))

        min_for_all_generations_for_mut_1_1_1 = np.empty((0, len(self.chromosoms) + 2))
        min_for_all_generations_for_mut_2_2_2 = np.empty((0, len(self.chromosoms) + 2))

        scores = []

        generation_index = 1

        while (generation_index-1) < self.n_generation:

            New_Population = np.empty((0, len(self.chromosoms)))

            All_in_Generation_X_1 = np.empty((0, len(self.chromosoms) + 1))
            All_in_Generation_X_2 = np.empty((0, len(self.chromosoms) + 1))

            Min_in_Generation_X_1 = []
            Min_in_Generation_X_2 = []

            Save_Best_Generation_X = np.empty((0, len(self.chromosoms) + 1))

            results_generation_X = []
            results_w_generation_X = []

            '''generation number'''
            Family = 1

            ''' family numbers '''
            for j in (range(int(self.n_population / 2))):

                Parents = np.empty((0, len(self.chromosoms)))

                for i in range(2):

                    Warrior_1_index = np.random.randint(0, len(n_list))

                    Warrior_2_index = np.random.randint(0, len(n_list))

                    Warrior_3_index = np.random.randint(0, len(n_list))

                    Warrior_4_index = np.random.randint(0, len(n_list))

                    while Warrior_1_index == Warrior_2_index:
                        Warrior_1_index = np.random.randint(0, len(n_list))

                    while Warrior_2_index == Warrior_3_index:
                        Warrior_3_index = np.random.randint(0, len(n_list))

                    while Warrior_1_index == Warrior_3_index:
                        Warrior_3_index = np.random.randint(0, len(n_list))

                    while Warrior_3_index == Warrior_4_index:
                        Warrior_4_index = np.random.randint(0, len(n_list))

                    Warrior_1 = n_list[Warrior_1_index]

                    Warrior_2 = n_list[Warrior_2_index]

                    Warrior_3 = n_list[Warrior_3_index]

                    Warrior_4 = n_list[Warrior_4_index]


                    Prize_Warrior_1 = self.fitness(mut_chroms=Warrior_1, X=self.X,Y=self.Y, model=self.model)

                    Prize_Warrior_2 = self.fitness(mut_chroms=Warrior_2, X=self.X,Y=self.Y, model=self.model)

                    Prize_Warrior_3 = self.fitness(mut_chroms=Warrior_3, X=self.X,Y=self.Y, model=self.model)

                    Prize_Warrior_4 = self.fitness(mut_chroms=Warrior_4, X=self.X, Y=self.Y, model=self.model)


                    if Prize_Warrior_1 == min(Prize_Warrior_1, Prize_Warrior_2, Prize_Warrior_3,Prize_Warrior_4):
                        Winner = Warrior_1
                        Prize = Prize_Warrior_1

                    if Prize_Warrior_2 == min(Prize_Warrior_1, Prize_Warrior_2, Prize_Warrior_3,Prize_Warrior_4):
                        Winner = Warrior_2
                        Prize = Prize_Warrior_2

                    if Prize_Warrior_3 == min(Prize_Warrior_1, Prize_Warrior_2, Prize_Warrior_3, Prize_Warrior_4):
                        Winner = Warrior_3
                        Prize = Prize_Warrior_3

                    if Prize_Warrior_4 == min(Prize_Warrior_1, Prize_Warrior_2, Prize_Warrior_3, Prize_Warrior_4):
                        Winner = Warrior_4
                        Prize = Prize_Warrior_4

                    Parents = np.vstack((Parents, Winner))


                Parent_1 = Parents[0]
                Parent_2 = Parents[1]

                # Crossover
                Child_1 = np.empty((0, len(self.chromosoms)))
                Child_2 = np.empty((0, len(self.chromosoms)))

                Ran_CO_1 = np.random.rand()

                if Ran_CO_1 < self.p_covering:

                    Cr_1 = np.random.randint(0, len(self.chromosoms))
                    Cr_2 = np.random.randint(0, len(self.chromosoms))

                    while Cr_1 == Cr_2:
                        Cr_2 = np.random.randint(0, len(self.chromosoms))

                    if Cr_1 < Cr_2:

                        Med_Seg_1 = Parent_1[Cr_1:Cr_2 + 1]
                        Med_Seg_2 = Parent_2[Cr_1:Cr_2 + 1]

                        First_Seg_1 = Parent_1[:Cr_1]
                        Sec_Seg_1 = Parent_1[Cr_2 + 1:]

                        First_Seg_2 = Parent_2[:Cr_1]
                        Sec_Seg_2 = Parent_2[Cr_2 + 1:]

                        Child_1 = np.concatenate((First_Seg_1, Med_Seg_2, Sec_Seg_1))
                        Child_2 = np.concatenate((First_Seg_2, Med_Seg_1, Sec_Seg_2))

                    else:
                        Med_Seg_1 = Parent_1[Cr_2:Cr_1 + 1]
                        Med_Seg_2 = Parent_2[Cr_2:Cr_1 + 1]

                        First_Seg_1 = Parent_1[:Cr_2]
                        Sec_Seg_1 = Parent_1[Cr_1 + 1:]

                        First_Seg_2 = Parent_2[:Cr_2]
                        Sec_Seg_2 = Parent_2[Cr_1 + 1:]

                        Child_1 = np.concatenate((First_Seg_1, Med_Seg_2, Sec_Seg_1))
                        Child_2 = np.concatenate((First_Seg_2, Med_Seg_1, Sec_Seg_2))

                else:
                    Child_1 = Parent_1
                    Child_2 = Parent_2


                Mutated_Child_1 = []
                Mutated_Child_1 = self._mutation(Child_1)

                Mutated_Child_2 = []
                Mutated_Child_2 = self._mutation(Child_2)

                # For mutated child
                OF_So_Far_M1 = self.fitness(mut_chroms=Mutated_Child_1,X=self.X,Y=self.Y, model=self.model)
                OF_So_Far_M2 = self.fitness(mut_chroms=Mutated_Child_2,X=self.X,Y=self.Y, model=self.model)

                # fitness scores per mutated child
                All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]

                All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_M1, All_in_Generation_X_1_1_temp))

                All_in_Generation_X_2_2_temp = Mutated_Child_2[np.newaxis]

                All_in_Generation_X_2_2 = np.column_stack((OF_So_Far_M2, All_in_Generation_X_2_2_temp))

                All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1, All_in_Generation_X_1_1))
                All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2, All_in_Generation_X_2_2))

                Save_Best_Generation_X = np.vstack((All_in_Generation_X_1, All_in_Generation_X_2))
                New_Population = np.vstack((New_Population, Mutated_Child_1, Mutated_Child_2))
                scores.append(np.array((All_in_Generation_X_1[:,0]+All_in_Generation_X_2[:,0])/2).astype(np.float)[0])

                R_1 = []
                R_1 = self.__optim_count(All_in_Generation_X_1)
                Min_in_Generation_X_1 = R_1[np.newaxis]

                R_2 = []
                R_2 = self.__optim_count(All_in_Generation_X_2)
                Min_in_Generation_X_2 = R_2[np.newaxis]

                Family = Family + 1

            t = 0
            R_11 = []
            R_11 = self.__optim_count(Save_Best_Generation_X)
            results_generation_X = R_11[np.newaxis]

            t = 0
            R_22 = []
            R_22 = self.__optim_count(Save_Best_Generation_X, maximize=True)

            results_w_generation_X = R_22[np.newaxis]

            Darwin_Gay = results_generation_X[:]
            No_So_Darwin_Gay = results_w_generation_X[:]

            Darwin_Gay = Darwin_Gay[0:, 1:].tolist()
            No_So_Darwin_Gay = No_So_Darwin_Gay[0:, 1:].tolist()

            Best_1 = np.where((New_Population == Darwin_Gay))
            Worst_1 = np.where((New_Population == No_So_Darwin_Gay))

            New_Population[Worst_1] = Darwin_Gay
            n_list = New_Population

            min_for_all_generations_for_mut_1 = np.vstack((min_for_all_generations_for_mut_1,
                                                          Min_in_Generation_X_1))

            min_for_all_generations_for_mut_2 = np.vstack((min_for_all_generations_for_mut_2,
                                                          Min_in_Generation_X_2))

            min_for_all_generations_for_mut_1_1 = np.insert(Min_in_Generation_X_1, 0, generation_index)
            min_for_all_generations_for_mut_2_2 = np.insert(Min_in_Generation_X_2, 0, generation_index)

            min_for_all_generations_for_mut_1_1_1 = np.vstack(
                (min_for_all_generations_for_mut_1_1, min_for_all_generations_for_mut_1_1_1))

            min_for_all_generations_for_mut_2_2_2 = np.vstack(
                (min_for_all_generations_for_mut_2_2, min_for_all_generations_for_mut_2_2_2))

            print("[FIT GENE: %0.0f, AV: %0.4f, MIN: %0.4f, SD: %0.4f, MAX: %0.4f]" \
                  % (generation_index, np.abs(np.mean(scores[-80:])), np.min(np.abs(scores[-80:])),
                     np.std(scores[-80:]), np.max(np.abs(scores[-80:]))))

            generation_index += 1


        one_final_guy = np.vstack((min_for_all_generations_for_mut_1_1_1,
                                   min_for_all_generations_for_mut_2_2_2))
        t = 0
        Final_Here = []
        for i in one_final_guy:
            if (one_final_guy[t, 1]) <= min(one_final_guy[:, 1]):
                Final_Here = one_final_guy[t, :]
            t = t + 1

        one_final_guy_final = Final_Here[np.newaxis]
        print()
        print("Min in All Generation", one_final_guy_final)
        print()
        print("Final Solution", one_final_guy_final[:, 2:])
        print()
        print("Highest Accuracy", 1-one_final_guy_final[:, 1])

        A11 = one_final_guy_final[:, 2:][0]
        t = 0
        self.emp_list = []
        for i in A11:
            if A11[t] == 1:
                self.emp_list.append(t)
            t = t + 1
        print()
        print("Features included:", self.emp_list)
        self.metrics(scores)

    @staticmethod
    def metrics(scores):
        percentileList = [i / (len(scores) - 1) for i in range(len(scores))]
        validationAccuracyList = list([lambda x: x.values, scores])
        import matplotlib.pyplot as plt
        from scipy import interpolate


        tck = interpolate.splrep(percentileList, validationAccuracyList[1], s=5.0)
        ynew = interpolate.splev(percentileList, tck)

        e = plt.figure(1)
        plt.plot(percentileList, validationAccuracyList[1], marker='o', color='r')
        plt.plot(percentileList, ynew, color='b')
        plt.title('Fitness with Cubic-Spline Interpolation')
        plt.xlabel('Population percentile')
        plt.ylabel('Error')
        plt.vlines(0.5, min(validationAccuracyList[1]), max(validationAccuracyList[1]), color="k", linestyles="--", lw=1)

        leg = plt.legend()
        leg.get_frame().set_alpha(0.4)
        plt.autoscale(tight=True)
        e.show()

        f = plt.figure(2)

        plt.scatter(percentileList, validationAccuracyList[1])
        plt.title('Fitness')
        plt.xlabel('Population percentile')
        plt.ylabel('Error')
        plt.vlines(0.5, min(validationAccuracyList[1]), max(validationAccuracyList[1]), color="k", linestyles="--", lw=1)

        leg = plt.legend()
        leg.get_frame().set_alpha(0.4)
        plt.autoscale(tight=True)
        f.show()
