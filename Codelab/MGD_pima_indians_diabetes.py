from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid (logistic) function
    :param x: array-like shape(n_sample, n_feature)
    :return: simgoid value (array like)
    """
    return 1. / (1. + np.exp(-x))

def load_data(file_name, names, preprocessing_type):
    data_frame = read_csv(file_name, names=names)
    array = data_frame.values

    # Separate array into input and output components
    X = array[:, 0:-1]
    y = array[:, -1]
    y = np.array([y]).T

    #preprosessing the data
    X_scaled = preprocess_data(X, preprocessing_type)
    return X_scaled, y


def preprocess_data(X, preprocessing_type):
    if preprocessing_type == "std":
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

    if preprocessing_type == "l1" or preprocessing_type == "l2":
        scaler = Normalizer(preprocessing_type)
        X_scaled = scaler.transform(X)

    if preprocessing_type == "min_max":
        # scaler = MinMaxScaler(preprocessing_type)
        # X_scaled = scaler.transform(X)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)

    # Summarize transformed data
    set_printoptions(precision=3)

    return X_scaled


def split_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, y_train, x_test, y_test

# def split_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0., random_state=42)
#     return X_train, X_test, y_train, y_test


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)
    for i in range(0, X.shape[0], minibatch_size):
        X_batch = X[i : i + minibatch_size, :]
        y_batch = y[i : i + minibatch_size]
        minibatches.append((X_batch, y_batch))
    return minibatches

#could add one more parameter: model
#h(x)
def logistic_val_func(theta, x):
    # forwarding
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))


def logistic_grad_func(theta, x, y):
    # TODO compute gradient
    y_hat = logistic_val_func(theta, x)
    x = np.c_[np.ones(x.shape[0]), x]
    grad = np.dot((y_hat - y).T, x)
    return grad


def logistic_cost_func(theta, x, y):
    # compute cost (loss)
    y_hat = logistic_val_func(theta, x)
    cost = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    cost *= -1.0 / x.shape[0]
    return cost

def mini_batch_grad_desc(x_train, y_train, theta, max_epoch, lr, batch_size, epsilon, momentum = None, adagrad = False):

    minibatches = get_minibatch(x_train, y_train, batch_size)
    x_mini, y_mini = minibatches[0]
    cost_iter = []
    cost = logistic_cost_func(theta, x_mini, y_mini)
    cost_iter.append([0, cost])
    itr = 1
    momentum_vector = [0.0]
    cost_change = 1
    while itr < max_epoch and cost_change > epsilon:
        pre_cost = cost
        for idx in range(0, len(minibatches)):
        #idx = np.random.randint(0, len(minibatches))
            x_mini, y_mini = minibatches[idx]
            grad = logistic_grad_func(theta, x_mini, y_mini)
            if momentum is not None:
                grad = momentum * momentum_vector[-1] - lr * grad * 1./ batch_size
                momentum_vector.append(grad)
                theta += grad
            else:
                theta -= lr * grad * 1. / batch_size


        #calculate cost with updated theta after each batch
        cost = logistic_cost_func(theta, x_mini, y_mini)
        cost_iter.append([itr, cost])
        cost_change = abs(cost - pre_cost)
        itr+=1

    return theta, cost_iter

    # prediction values
def pred_val(theta, X, hard=True):
    pred_prob = logistic_val_func(theta, X)
    pred_value = np.where(pred_prob > 0.5, 1, 0)
    if hard:
        return pred_value
    else:
        return pred_prob

def accuracy(theta, x_test, y_test):
    n_samples = x_test.shape[0]
    preds = pred_val(theta, x_test)
    acc = np.sum(1.0 * (preds == y_test)) / n_samples
    return acc


if __name__ == '__main__':
    #print('\nRescale using sklearn')
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    X, y = load_data(filename, names, "l2")
    x_train, y_train, x_test, y_test = split_data(X, y)

    max_epoch = 1000
    epsilon = 1e-3

    #Weight Initialization
    #np.random.seed(0)
    theta_1 = np.random.randn(1, x_train.shape[1] + 1)
    theta_2 = np.random.randn(1, x_train.shape[1] + 1)

    lr = 0.001
    cost_iter = []
    batch_size = 10

    # Minibatch Gradient Descent
    fitted_theta_1, cost_iter__1 = mini_batch_grad_desc(x_train, y_train, theta_1, max_epoch, lr, batch_size, epsilon, momentum=None, adagrad=False)
    acc1 = accuracy(fitted_theta_1, x_test, y_test)
    print("Minibatch Gradient Descent Accuracy: {}".format(acc1))

    # Minibatch Gradient Descent with Momentum
    fitted_theta_2, cost_iter_2 = mini_batch_grad_desc(x_train, y_train, theta_2, max_epoch, lr, batch_size, epsilon,
                                                   momentum=0.9, adagrad=False)
    acc2 = accuracy(fitted_theta_2, x_test, y_test)
    print("Minibatch Gradient Descent with Momentum Accuracy: {}".format(acc2))

    # Plot Testing
    #plot_line(x_test, y_test, theta=fitted_theta)