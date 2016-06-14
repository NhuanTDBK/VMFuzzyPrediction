# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from estimators.ACOEstimator import ACOEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.GAEstimator import GAEstimator
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from utils.GraphUtil import *
from utils.initializer import *
from estimators.FuzzyStep import *
from utils.SlidingWindowUtil import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.grid_search import delayed, Parallel


scaler = MinMaxScaler()
print 'FACO Model'
dat_usage = pd.read_csv('sample_610_10min.csv')['cpu_rate'][:2400]
# dat_usage = pd.read_csv('cpu_usage.csv')['cpu_usage']

# define training_set and testing _set
training_set_size = 1000
testing_set_size = 500


######## FUZZY before TRAINING
# length of sliding windows for input

#define training_set, testing set
X_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    X_train[i] = dat_usage[i]

y_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    y_train[i] = dat_usage[i + 1]

X_test = np.zeros(testing_set_size)
for i in range (0,testing_set_size):
    X_test[i] = dat_usage[i + training_set_size]

y_test = np.zeros(testing_set_size - 1)
for i in range (0,testing_set_size - 1):
    y_test[i] = dat_usage[i + training_set_size + 1]

# Number of hiddens node (one hidden layer)
n_hidden = 300

# Define neural shape
    # Input layer: [n_sample*n_size]
    # Hidden layer:
    # Output layer: regression
# neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]

# <codecell>

# fuzzy before training
# fuzzy step:
# fuzzy_set = fuzzy(X_train)[1]
# sliding_number = 4
def gridSearchSliding(sliding_number):
    score_min = 100
    print "Begin grid search..."
    for loop in np.arange(1):
        X_train_f = np.array(fuzzy(X_train))
        y_train_f = np.array(fuzzy(y_train))[sliding_number-1:]
        X_test_f = np.array(fuzzy(X_test))


        X_train_f = np.array(list(SlidingWindow(X_train_f,sliding_number,concatenate=True)))
        # y_train_f = np.array(fuzzy(y_train)[n_sliding_window-1:])
        X_test_f = np.array(list(SlidingWindow(X_test_f,sliding_number,concatenate=True)))
        # X_train_f, a = fuzzy(X_train,automf=True)
        # y_train_f,b = fuzzy(y_train,automf=True)
        # X_test_f,c = fuzzy(X_test,automf=True)
        neural_shape = [len(X_train_f[0]),n_hidden,len(y_train_f[0])]
        # Initialize neural network model for regression

        # <codecell>

        # Initialize ACO Estimator
        #estimator = ACOEstimator(Q=0.7,epsilon=0.1,number_of_solutions=130)
	estimator = GAEstimator(cross_rate=0.65, mutation_rate=0.01,pop_size=45)
        # estimator = GAEstimator(cross_rate=0.65,mutation_rate=0.01)
        fit_param = {'neural_shape':neural_shape}
        neuralNet = NeuralFlowRegressor(learning_rate=1E-03,verbose=1,steps=1000,activation='sigmoid')

        # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
        optimizer = OptimizerNNEstimator(estimator,neuralNet)
        # optimizer = neuralNet
        optimizer.fit(X_train_f,y_train_f,**fit_param)

        # <codecell>

        print  optimizer.score(X_train_f,y_train_f)
        y_pred = optimizer.predict(X_test_f)
        # #defuzzy step:
        ft = abs(defuzzy(X_test[sliding_number-1:],y_pred))
        #mean_squared_error
        score_fuzz = mean_squared_error(y_test[sliding_number-1:], ft)
        if(score_min > score_fuzz):
            score_min = score_fuzz
    print "Score of this sliding %s"%score_fuzz
    np.savez('model_saved/FACO_%s_sliding_%s'%(score_fuzz,sliding_number),y_pred=ft[:190],y_test=y_test[:190])
    return sliding_number,score_fuzz
print "Started to parallel..."
out = Parallel(n_jobs=-1)(delayed(gridSearchSliding)(i) for i in np.arange(2,11))
pd.DataFrame(out,columns=["sliding","score"]).to_csv("gridSearchSliding.csv",index=None)
#plot graph
#%matplotlib
#plot_figure(ft[:190],y_test[:190],color=['blue','red'],title='FACONN with sliding window = %s,hidden nodes = %s => rmse = %s'%(sliding_number,n_hidden,score_fuzz))

# <codecell>