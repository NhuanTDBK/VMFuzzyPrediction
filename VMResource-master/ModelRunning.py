from __init__ import *
from sklearn.metrics import mean_squared_error
from io_utils.GFeeder import GFeeder
from math import fabs
import skflow
from utils.GraphUtil import *
from io_utils.NumLoad import *
model = skflow.TensorFlowEstimator.restore("experiments/params/model_full_metric/")

dataFeeder = GFeeder(file_name='data/gdata/gcluster_1268205_1min.json')
# dataFeederNormalize = GFeeder()
#
metrics_types = [dataFeeder.CPU_UTIL,dataFeeder.DISK_IO_TIME,dataFeeder.DISK_SPACE,dataFeeder.MEM_USAGE]

# n_sliding_window = 2
# X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metrics=metrics_types,n_sliding_window=n_sliding_window)
# X_trainn,y_trainn,X_testn,y_testn = dataFeederNormalize.split_train_and_test(metrics=metrics_types,n_sliding_window=n_sliding_window)
X_trainn,y_trainn,X_testn,y_testn = load_training_from_npz("data/gdata/data_training.npz")
X_train,y_train,X_test,y_test = load_training_from_npz("data/gdata/data_training_origin.npz")
y_pred = model.predict(X_testn).tolist()
print mean_squared_error(y_pred,y_testn)
# plot_metric_figure(y_pred=y_pred,y_test=y_testn,metric_type=metrics_types)
for i in [1,2]:
    io_max = y_test[:,i].max()
    io_min = y_test[:,i].min()
    y_pred[i] = np.array(y_pred[i])*fabs(io_max-io_min)+io_min
# plot_figure(y_pred_convert, y_test[:, 1], title="GABP")