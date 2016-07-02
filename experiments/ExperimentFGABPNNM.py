from io_utils.NumLoad import *
from sklearn.cross_validation import train_test_split
from estimators.FuzzyFlow import FuzzyFlow
from utils.TrainingTestMaker import TrainingTestMaker
from scaling.ProactiveSLA import ProactiveSLA
from __init__fuzzy import *

dataset_holder = []
trainee_holder = {}
metrics = ["cpu_rate","mem_usage"]
arr_desk = ['X_train','y_train','X_test']
sliding_number = 3
data = pd.read_csv('sampling_617685_metric_10min_datetime.csv',parse_dates=True,index_col=0)[:3000]
def experiment(sliding_number):
    for metric in metrics:
        dat = pd.Series(data[metric].round(3))
        fuzzy_engine = FuzzyFlow()
        data_maker = TrainingTestMaker()
        fuzzy_engine.fit_transform(dat)
        sliding = np.array(list(SlidingWindow(fuzzy_engine.u_class_transform, sliding_number)))
        X_train, y_train, X_test, y_test = data_maker.make_fuzzy_test(sliding, fuzzy_engine.u_class_transform, dat)
        dataset_holder.append(fuzzy_engine)
        trainee_holder[metric] = {
            'X_train': X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
    y_train = np.asarray(zip(trainee_holder['cpu_rate']['y_train'], trainee_holder['mem_usage']['y_train']))
    # X_test = zip(trainee_holder['cpu_rate']['X_test'],trainee_holder['mem_usage']['X_test'])
    X_train = []
    X_test = []
    # y_train = []
    for i in np.arange(len(trainee_holder['cpu_rate']['X_train'])):
        tmp = zip(trainee_holder['cpu_rate']['X_train'][i], trainee_holder['mem_usage']['X_train'][i])
        X_train.append(np.ravel(tmp))
    for i in np.arange(len(trainee_holder['cpu_rate']['X_test'])):
        tmp = zip(trainee_holder['cpu_rate']['X_test'][i], trainee_holder['mem_usage']['X_test'][i])
        X_test.append(np.ravel(tmp))
    X_train = np.array(X_train)
    n_hidden = 10
    X_test = np.array(X_test)
    fit_params = {
        'neural_shape': [len(X_train[0]), n_hidden, 2]
    }
    ga_estimator = GAEstimator(cross_rate=0.5, mutation_rate=0.06, gen_size=100, pop_size=30)
    nn = NeuralFlowRegressor(hidden_nodes=[n_hidden], optimize='Adam'
                             , steps=7000, learning_rate=1E-02)
    classifier = OptimizerNNEstimator(ga_estimator, nn)
    classifier.fit(X_train,y_train, **fit_params)
    y_pred = np.round(abs(classifier.predict(X_test)))
    y_cpu = dataset_holder[0].inverse_transform(abs(y_pred[:, 0]))
    y_ram = dataset_holder[1].inverse_transform(abs(y_pred[:, 1]))
    score_mae_CPU = mean_absolute_error(y_cpu, trainee_holder['cpu_rate']['y_test'])
    score_mae_RAM = mean_absolute_error(y_ram, trainee_holder['mem_usage']['y_test'])
    y_test = zip(trainee_holder['cpu_rate']['y_test'], trainee_holder['mem_usage']['y_test'])
    np.savez('model_saved/FGABPNNM_%s_%s' % (sliding_number, score_mae_CPU), y_pred=y_pred, y_true=y_test)
    return sliding_number, score_mae_CPU, score_mae_RAM
result = [experiment(sliding_number=i) for i in np.arange(2, 6)]
result = pd.DataFrame(result, columns=["sliding_number", "MAE CPU", "MAE RAM"])
result.to_csv('fuzzy_gabpnn_experimentm.csv')

