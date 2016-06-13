from sklearn.grid_search import GridSearchCV

from estimators.ACOEstimator import ACOEstimator
from io_utils.GFeeder import GFeeder
from utils.initializer import *

param_dicts = {
    "Q":np.arange(0.01,0.1,step=0.01),
    "epsilon":np.arange(0.1,1.0,step=0.05),
    "number_of_solutions":np.arange(30,200)
}
n_windows = 4
n_hidden = 10
# range_training = (-1,28919)
# range_test = (28919,-1)
# metric_types = ["cpu_util","disk_write_rate","disk_read_rate","network_"]
dataFeeder = GFeeder()
X_train,y_train, X_test,y_test = dataFeeder.split_train_and_test(n_sliding_window=n_windows)

neural_shape = [n_windows*len(dataFeeder.metric_type),n_hidden,len(dataFeeder.metric_type)]
estimator = ACOEstimator()
archive_solution = construct_solution(estimator.number_of_solutions,neural_shape)
fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)
