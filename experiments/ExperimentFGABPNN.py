import math

from pandas import DataFrame

from __init__fuzzy import *


def experiment(sliding_number=3, hidden_node=15):
    # result = []
    print "New loop"
    X_train_size = int(len(u_class_transform) * 0.7)
    sliding = np.array(list(SlidingWindow(u_class_transform, sliding_number)))
    sliding = np.array(sliding, dtype=np.int32)
    X_train = sliding[:X_train_size]
    y_train = u_class_transform[sliding_number:X_train_size + sliding_number]
    X_test = sliding[X_train_size:]
    y_test = u_class_transform[X_train_size + sliding_number - 1:]
    y_actual_test = dat[X_train_size + sliding_number - 1:].tolist()
    # # Define classifier
    n_hidden = len(X_train[0]) + sliding_number

    fit_params = {
        'neural_shape': [len(X_train[0]), n_hidden, 1]
    }
    ga_estimator = GAEstimator(cross_rate=0.15, mutation_rate=0.06, gen_size=100, pop_size=30)
    nn = NeuralFlowRegressor(hidden_nodes=[n_hidden], optimize='Adam'
                             , steps=20000, learning_rate=1E-02)
    classifier = OptimizerNNEstimator(ga_estimator, nn)

    a = classifier.fit(X_train, y_train, **fit_params)
    ypred = np.round(abs(classifier.predict(X_test))).flatten()
    ypred_defuzzy = [defuzzy(item % len(u_unique_mapping), u_unique_mapping, u_midpoints) for item in ypred]
    score_mape = mean_absolute_error(ypred_defuzzy, y_actual_test)
    score_rmse = math.sqrt(mean_squared_error(ypred_defuzzy, y_actual_test))

    return sliding_number, score_rmse, score_mape


result = [experiment(sliding_number=i) for i in np.arange(2, 6)]
result = DataFrame(result, columns=["sliding_number", "rmse", "mae"])
result.to_csv('fuzzy_gabpnn_experiment.csv')