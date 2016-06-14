
# coding: utf-8

# In[12]:

import datetime

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid,Parallel,delayed

from estimators.NeuralFlow import NeuralFlowRegressor
from io.MetricFeeder import MetricFeeder

#----------------
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
params_estimate = {
    "n_windows":np.arange(5,20),
    "hidden_node":np.arange(10,40)
}
result = {}
candidate_param = ParameterGrid(param_grid=params_estimate)
dataFeeder = MetricFeeder(split_size=5)
def get_data(n_windows):
    return (n_windows,dataFeeder.split_train_and_test(metric_types,n_windows))
print "Getting data"
tmp = Parallel(n_jobs=-1)(delayed(get_data)(n_windows) for n_windows in params_estimate["n_windows"])
data_train = dict((x,y) for x,y in tmp)
print data_train.keys()
def estimator(n_windows,n_hidden_nodes):
    data = data_train[n_windows]
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]

    fit_param = {
                'neural_shape':[2*n_windows,n_hidden_nodes,2]
            }
    neuralNet = NeuralFlowRegressor()
    kfold = KFold(X_train.shape[0],5)
    score_lst = np.zeros(len(kfold),dtype=np.float32)
    for k,(train,test) in enumerate(kfold):
        neuralNet.fit(X_train[train],y_train[train],**fit_param)
    nn_shape = "%s-%s"%(2*n_windows,n_hidden_nodes)
    score = neuralNet.score(X_test,y_test)
    neuralNet.save("tmp/score_%s"%score)
    return nn_shape,score

# print neuralNet.score(X_test,y_test)
# y_pred = neuralNet.predict(X_test)
# plot_figure(y_pred[:,0],y_test[:,0])
result = [Parallel(n_jobs=-1)(delayed(estimator)(k["n_windows"],k["hidden_node"]) for k in candidate_param)]
np.savez("result_model_%s"%datetime.datetime.now(),result=result)
