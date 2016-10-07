from __init__fuzzy import *

dat = pd.read_csv('sampling_617685_metric_10min_datetime.csv',parse_dates=True,index_col=0)[:3000]
gFeeder = GFeeder()
metrics = ['cpu_rate','mem_usage','disk_space']
def experiment(sliding_number):
    X_train, y_train, X_test, y_test = gFeeder.split_train_and_test(dat, metrics=metrics, n_sliding_window=sliding_number)
    classifier = KerasRegressor(hidden_nodes=[32],steps=7000,learning_rate=1E-02)
    a = classifier.fit(X_train, y_train)
    ypred = classifier.predict(X_test)
    y_cpu = ypred[:,0]
    y_ram = ypred[:,1]
    y_disk_space = ypred[:,2]
    score_mae_CPU = mean_absolute_error(y_cpu, y_test[:,0])
    score_mae_RAM =  mean_absolute_error(y_ram, y_test[:,1])
    score_mae_diskio = mean_absolute_error(y_disk_space,y_test[:,2])
    np.savez('model_saved/BPNNM_%s_%s'%(sliding_number,score_mae_CPU),y_pred=ypred, y_true=y_test)
    return sliding_number, score_mae_CPU, score_mae_RAM, score_mae_diskio

result = [[experiment(sliding_number=i) for i in np.arange(2,6)] for j in np.arange(10)]
cols = ["sliding_number"]
cols.extend(metrics)
results = pd.DataFrame(np.array(result).reshape(-1,3), columns=cols)
results.to_csv('experiment_logs/bpnnm_experiment.csv')

