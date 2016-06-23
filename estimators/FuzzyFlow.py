import pandas as pd
import numpy as np
import math
class FuzzyFlow(object):
    def __init__(self,fuzzy_distance = 0.02,fuzzy_set_size = 0, fuzzy_norm = 0.25):

        self.fuzzy_distance = fuzzy_distance
        self.fuzzy_set_size = fuzzy_set_size
        self.fuzzy_norm = fuzzy_norm
    def fuzzy(self,training_set):
        length_x_train = training_set.size
        #  Calculate difference in training data
        difference = np.zeros(length_x_train - 1)
        for i in range(0, length_x_train - 1):
            difference[i] = training_set[i + 1] - training_set[i] + self.fuzzy_norm
        # Estimate fuzzy size
        if(self.fuzzy_set_size==0):
            self.fuzzy_set_size = int(math.ceil(difference.max()/self.fuzzy_distance)+1)
        print self.fuzzy_set_size
        # Generate fuzzy set
        self.fuzzy_set = np.zeros(self.fuzzy_set_size)
        for i in range(0, self.fuzzy_set_size):
            self.fuzzy_set[i] = self.fuzzy_distance * (i + 0.5)
        fuzzy_result = np.zeros([length_x_train - 1, self.fuzzy_set_size])
        for i in range(1, length_x_train - 2):
            j = int(difference[i] / self.fuzzy_distance)
            fuzzy_result[i][j] = 1
            fuzzy_result[i][j - 1] = (self.fuzzy_set[j + 1] + self.fuzzy_set[j] - 2 * difference[i]) / (2 * self.fuzzy_distance)
            fuzzy_result[i][j + 1] = (- self.fuzzy_set[j - 1] - self.fuzzy_set[j] + 2 * difference[i]) / (2 * self.fuzzy_distance)
        # df = pd.DataFrame(data = fuzzy_result)
        # df.to_csv('x_train.csv')
        self.difference = difference
        return fuzzy_result
    def defuzzy(self,testing_set, training_result):
        # df = pd.DataFrame(data=training_result)
        # df.to_csv('y_predict.csv')
        # difference = self.difference
        length_y_test = testing_set.size
        y_pred = np.zeros([length_y_test - 1])
        for i in range (0, length_y_test - 1):
            tu = 0
            mau = 0

            for j in range (0, self.fuzzy_set_size):
                tu = tu + self.fuzzy_set[j] *  training_result[i][j]
                mau = mau + training_result[i][j]
            difference = tu / mau - self.fuzzy_norm
            y_pred[i] = testing_set[i] + difference
        return y_pred
    def fit_transform(self,data):
        return self.fuzzy(training_set=data)
    def inverse_transform(self, testing_set, training_result):
        return self.defuzzy(testing_set=testing_set,training_result=training_result)