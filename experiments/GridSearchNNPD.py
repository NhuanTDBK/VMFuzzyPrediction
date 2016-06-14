# Experiment GABPNN
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from estimators.ACOEstimator import ACOEstimator
from io_utils.GFeeder import GFeeder
from utils.GraphUtil import *
from utils.initializer import *
from utils.SlidingWindowUtil import SlidingWindow
from sklearn.metrics.regression import mean_squared_error,mean_absolute_error
# length of sliding windows for input
n_sliding_window = 3
dat_usage = pd.read_csv('../sample_610_10min.csv')['cpu_rate'][:2400]
# dat_usage = pd.read_csv('cpu_usage.csv')['cpu_usage']

# define training_set and testing _set
training_set_size = 1000
testing_set_size = 500
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

X_train_f = np.array(list(SlidingWindow(X_train,n_sliding_window,concatenate=False)))

X_test_f = np.array(list(SlidingWindow(X_test,n_sliding_window,concatenate=False)))
#Getting Google cluster data
dataFeeder = GFeeder(skip_lists=3)
metrics_types = ['cpu_rate']
print "Getting data"
# X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(data=dat_usage,metrics=metrics_types,n_sliding_window=n_sliding_window,train_size=0.7)
# Number of hiddens node (one hidden layer)
n_hidden = 15
neural_shape = [X_train_f.shape[1],n_hidden,1]
fit_param = {'neural_shape':neural_shape}

neuralNet = NeuralFlowRegressor(learning_rate= 1E-02)
# estimator = GAEstimator(cross_rate=0.7, mutation_rate=0.01)
estimator = ACOEstimator(Q=0.7,epsilon=0.1,number_of_solutions=50)
optimizer = OptimizerNNEstimator(estimator,neuralNet)
optimizer.fit(X_train_f,y_train[n_sliding_window-1:],**fit_param)
y_pred = optimizer.predict(X_test_f)
# score_nn =  optimizer.score(X_test_f,y_test[n_sliding_window-2:])
score_nn = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_test[n_sliding_window-2:]))
print score_nn
np.savez('GABPNN_%s_sliding_%s'%(score_nn,n_sliding_window),y_test=y_test, y_pred=y_pred)
plot_figure(y_pred=y_pred,y_true=y_test[1:],title="BPNN Sliding window = %s, rmse = %s, hidden nodes = %s "%(n_sliding_window,score_nn,n_hidden))
# score_list = {}
# for n_hidden in np.arange(240,300,step=1):
#     # n_hidden = 80
#     # Define neural shape
#         # Input layer: [n_sample*n_size]
#         # Hidden layer:
#         # Output layer: regression
#     neural_shape = [dataFeeder.input_size,n_hidden,dataFeeder.output_size]
#     # Initialize GA Estimator
#     # estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.04,pop_size=60,gen_size=100)
#
#     fit_param = {'neural_shape':neural_shape}
#
#     # Initialize neural network model for regression
#     neuralNet = NeuralFlowRegressor()
#
#     # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
#     # optimizer = OptimizerNNEstimator(estimator,neuralNet)
#     optimizer = neuralNet
#     optimizer.fit(X_train,y_train,**fit_param)
#     score = optimizer.score(X_test,y_test)
#     print score
#     score_list[n_hidden]=score
#     optimizer.save("params/model_full_metric_%s"%score)
# # if score < 0.01:
# # y_pred = optimizer.predict(X_test)
# # plot_metric_figure(y_pred=y_pred,y_test=y_test, metric_type=dataFeeder.metrics,title="GANN")
# # plot_metric_figure(y_pred=y_pred,y_test=y_test,metric_type=metrics_types,title=" GANN ")
# #optimizer.save("params/model_full_metric_%s"%score)
# score_list = pd.Series(score_list)
# print "Optimal hidden nodes: %s, with score = %s"%(score_list.argmin(),score_list.min())
