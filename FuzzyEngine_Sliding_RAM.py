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
from estimators.FuzzyFlow import *
from utils.SlidingWindowUtil import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from metrics.mean_absolute_percentage_error import mean_absolute_percentage_error

# <codecell>

scaler = MinMaxScaler()
print 'FACO Model'
# dat_usage = pd.read_csv('sample_610_10min.csv')['cpu_rate'][:2400]
# dat_usage = pd.read_csv('cpu_usage.csv')['cpu_usage']
dat_usage = pd.Series(scaler.fit_transform(pd.read_csv('sample_610_10min.csv')['mem_usage'][:2400]))
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
n_hidden = 600
sliding_number = 3

# <codecell>

fuzzyEngine = FuzzyFlow(fuzzy_distance=0.026,fuzzy_set_size=26)
X_train_f = np.array(fuzzyEngine.fit_transform(X_train))
y_train_f = np.array(fuzzyEngine.fit_transform(y_train))[sliding_number-1:]
X_test_f = np.array(fuzzyEngine.fit_transform(X_test))


X_train_f = np.array(list(SlidingWindow(X_train_f,sliding_number,concatenate=True)))
# y_train_f = np.array(fuzzy(y_train)[n_sliding_window-1:])
X_test_f = np.array(list(SlidingWindow(X_test_f,sliding_number,concatenate=True)))
# X_train_f, a = fuzzy(X_train,automf=True)
# y_train_f,b = fuzzy(y_train,automf=True)
# X_test_f,c = fuzzy(X_test,automf=True)
neural_shape = [len(X_train_f[0]),n_hidden,len(y_train_f[0])]
# Initialize neural network model for regression

# <codecell>


# <codecell>

# Initialize ACO Estimator
estimator = ACOEstimator(Q=0.5,epsilon=0.1,number_of_solutions=100)
# estimator = GAEstimator(cross_rate=0.7, mutation_rate=0.01,pop_size=45)
# estimator = GAEstimator(cross_rate=0.65,mutation_rate=0.01)
fit_param = {'neural_shape':neural_shape}
neuralNet = NeuralFlowRegressor(learning_rate=1E-03,verbose=1,steps=6000)

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
score_mae = mean_absolute_error(y_test[sliding_number-1:], ft)
# score_mape = mean_absolute_percentage_error(y_pred=ft,y_true=y_test[sliding_number-1:])
print "Score rmse = %s, mae = %s"%(score_fuzz,score_mae)
np.savez('Fuzzy_Time_Series_%s_sliding_%s'%(score_fuzz,sliding_number),y_test=y_test[:190], y_pred=ft[:190])
#plot graph
#%matplotlib
plot_figure(ft[:190],y_test[:190],color=['blue','red'],title='Fuzzy Time Series for RAM usage with sliding window = %s,hidden nodes = %s => rmse = %s'%(sliding_number,n_hidden,score_fuzz))
# <codecell>

