# Experiment GABPNN
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from io_utils.GFeeder import GFeeder
from utils.GraphUtil import *
from utils.initializer import *
from estimators.FuzzyStep import *


print "Getting data"



data = pd.read_csv('../cpu_usage.csv')

# define training_set and testing _set
training_set_size = 2000
testing_set_size = 500


######## FUZZY before TRAINING
# length of sliding windows for input
n_sliding_window = 1
X_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    X_train[i] = data.cpu_usage[i]

y_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    y_train[i] = data.cpu_usage[i + 1]

X_test = np.zeros(testing_set_size)
for i in range (0,testing_set_size):
    X_test[i] = data.cpu_usage[i + training_set_size]

y_test = np.zeros(testing_set_size - 1)
for i in range (0,testing_set_size - 1):
    y_test[i] = data.cpu_usage[i + training_set_size + 1]

# Number of hiddens node (one hidden layer)
n_hidden = 52

# Define neural shape
    # Input layer: [n_sample*n_size]
    # Hidden layer:
    # Output layer: regression
# neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]
neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]

# Initialize GA Estimator
estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.04,pop_size=60,gen_size=100)

fit_param = {'neural_shape':neural_shape}


# fuzzy before training
# fuzzy step:
X_train_f = fuzzy(X_train)
y_train_f = fuzzy(y_train)
X_test_f = fuzzy(X_test)

# Initialize neural network model for regression
neuralNet = NeuralFlowRegressor()

# There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
optimizer = OptimizerNNEstimator(estimator,neuralNet)
# optimizer = neuralNet
optimizer.fit(X_train_f,y_train_f,**fit_param)
print  optimizer.score(X_train_f,y_train_f)

y_pred = optimizer.predict(X_test_f)

#defuzzy step:
ft = defuzzy(X_test, y_pred)
#mean_squared_error
print mean_squared_error(y_test, ft)
#plot graph
plot_figure(ft,y_test,color=['blue','red'],title='FGANN')





# ######## NORMAL, TRAINING without FUZZY
# # length of sliding windows for input
# n_sliding_window = 3
# X_train = np.zeros([training_set_size,n_sliding_window])
# for i in range (0,training_set_size):
#     for j in range (0, n_sliding_window):
#         X_train[i][j] = data.cpu_usage[i+j]
#
# y_train = np.zeros(training_set_size)
# for i in range (0,training_set_size):
#     y_train[i] = data.cpu_usage[i + n_sliding_window]
#
# X_test = np.zeros([testing_set_size, n_sliding_window])
# for i in range (0,testing_set_size):
#     for j in range(0, n_sliding_window):
#         X_test[i][j] = data.cpu_usage[i + training_set_size + j]
#
# y_test = np.zeros(testing_set_size )
# for i in range (0,testing_set_size ):
#     y_test[i] = data.cpu_usage[i + training_set_size + n_sliding_window]
#
#
# n_hidden = 25
#
# # Define neural shape
#     # Input layer: [n_sample*n_size]
#     # Hidden layer:
#     # Output layer: regression
# # neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]
# neural_shape = [n_sliding_window,n_hidden,1]
#
# # Initialize GA Estimator
# estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.04,pop_size=60,gen_size=100)
#
# fit_param = {'neural_shape':neural_shape}
#
# # Initialize neural network model for regression
# neuralNet = NeuralFlowRegressor()
#
# # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
# optimizer = OptimizerNNEstimator(estimator,neuralNet)
# # optimizer = neuralNet
# optimizer.fit(X_train,y_train,**fit_param)
# print  optimizer.score(X_train,y_train)
# # print score
# # score_list[n_hidden]=score
# # # if score < 0.01:
#
# y_pred = optimizer.predict(X_test)
#
# #mean_squared_error
# print mean_squared_error(y_test, y_pred)
# #plot graph
# plot_figure(y_pred,y_test,color=['blue','red'],title='FGANN')