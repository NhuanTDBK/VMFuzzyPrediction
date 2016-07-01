from __init__fuzzy import *

dat = pd.read_csv('../sampling_617685_metric_10min_datetime.csv',parse_dates=True,index_col=0)[:3000]
gFeeder = GFeeder()
def experiment(sliding_number):
    X_train, y_train, X_test, y_test = gFeeder.split_train_and_test(dat, metrics=['cpu_rate','mem_usage'], n_sliding_window=sliding_number)
    n_hidden = 10
    fit_params = {
        'neural_shape': [len(X_train[0]), n_hidden, 2]
    }
    ga_estimator = GAEstimator(cross_rate=0.5, mutation_rate=0.06, gen_size=100, pop_size=30)
    nn = NeuralFlowRegressor(hidden_nodes=[n_hidden], optimize='Adam'
                             , steps=7000, learning_rate=1E-02)
    classifier = OptimizerNNEstimator(ga_estimator, nn)
    a = classifier.fit(X_train, y_train, **fit_params)
    ypred = classifier.predict(X_test)
    y_cpu = ypred[:,0]
    y_ram = ypred[:,1]
    score_mae_CPU = mean_absolute_error(y_cpu, y_test[:,0])
    score_mae_RAM =  mean_absolute_error(y_ram, y_test[:,1])
    np.savez('GABPNNM_%s_%s'%(sliding_number,score_mae_CPU),y_pred=ypred, y_true=y_test)
    return sliding_number, score_mae_CPU, score_mae_RAM

result = [experiment(sliding_number=i) for i in np.arange(2, 6)]
result = pd.DataFrame(result, columns=["sliding_number", "MAE CPU", "MAE RAM"])
result.to_csv('gabpnnm_experiment.csv')

