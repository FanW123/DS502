from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_data(file_name, names, preprocessing_type):
    data_frame = read_csv(file_name, names=names)
    array = data_frame.values
    # Separate array into input and output components
    X = array[:, 0:-1]
    y = array[:, -1]
    #change y from shape(768, ) tp (768,1)
    y = np.array([y]).T
    #preprosessing the data
    X_scaled = preprocess_data(data_set = X, type = preprocessing_type)
    return X_scaled, y


def preprocess_data(data_set, type):
    if type == "std":
        scaler = StandardScaler().fit(data_set)
        X_scaled = scaler.transform(data_set)
    if type == "l1" or type == "l2":
        scaler = Normalizer(norm = type)
        X_scaled = scaler.fit_transform(data_set)
    if type == "min_max":
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(data_set)
    # Summarize transformed data
    set_printoptions(precision=3)

    return X_scaled


# def split_data(X, y, train_percent=0.7):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_percent, random_state=29)
#     #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
#     return x_train, y_train, x_test, y_test

def split_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    return x_train, y_train, x_test, y_test


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)
    for i in range(0, X.shape[0], minibatch_size):
        X_batch = X[i : i + minibatch_size, :]
        y_batch = y[i : i + minibatch_size]
        minibatches.append((X_batch, y_batch))
    return minibatches


def sigmoid(x):
    """
    Sigmoid (logistic) function
    :param x: array-like shape(n_sample, n_feature)
    :return: simgoid value (array like)
    """
    return 1. / (1. + np.exp(-x))

def logistic_val_func(theta, x):
    #return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))

def logistic_cost_func(theta, x, y):
    # compute cost (loss)
    y_hat = logistic_val_func(theta, x)
    cost = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    cost *= -1.0 / x.shape[0]
    return cost

def logistic_grad_func(theta, x, y):
    # TODO compute gradient
    y_hat = logistic_val_func(theta, x)
    x = np.c_[np.ones(x.shape[0]), x]
    grad = np.dot((y_hat - y).T, x)
    return grad

def mini_batch_grad_desc(x_train, y_train, theta, max_epoch = 100, lr = 0.001, batch_size = 10, epsilon = 1e-5, momentum = None, adagrad = False):

    minibatches = get_minibatch(x_train, y_train, batch_size)
    x_mini, y_mini = minibatches[0]
    cost_iter = []
    cost = logistic_cost_func(theta, x_mini, y_mini)
    cost_iter.append(cost)
    itr = 1
    momentum_vector = [0.0]
    cost_change = 1
    #m = x_train.shape[1] + 1
    #hisgrad = [np.zeros(m)]
    hisgrad = 0
    while itr < max_epoch and cost_change > epsilon:
        pre_cost = cost
        prev_theta = 0
        for idx in range(0, len(minibatches)):
            x_mini, y_mini = minibatches[idx]
            grad = logistic_grad_func(theta, x_mini, y_mini)
            hisgrad += grad**2
            if momentum is not None:
                grad = momentum * momentum_vector[-1] - lr * grad * 1./ batch_size
                momentum_vector.append(grad)
                theta += grad
            elif adagrad:
               grad = lr * grad * 1./ (np.sqrt(hisgrad * batch_size))
               theta -= grad
            else:
                theta -= lr * grad * 1. / batch_size


        #calculate cost with updated theta after each batch
            cost = logistic_cost_func(theta, x_mini, y_mini)
            cost_iter.append(cost)
            cost_change = abs(cost - pre_cost)
            if cost_change < epsilon:
                break
        itr+=1
    print("itr: {}, cost: {}". format(itr, cost))
    return theta, np.array(cost_iter)

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

    #Weight Initialization
    #np.random.seed(0)
    theta = np.random.randn(1, x_train.shape[1] + 1)

    lr = 0.001
    batch_size = 10

    # Minibatch Gradient Descent
    fitted_theta_1, cost_iter_1 = mini_batch_grad_desc(x_train, y_train, theta)
    acc1 = accuracy(fitted_theta_1, x_test, y_test)
    print("Minibatch Gradient Descent Accuracy: {}".format(acc1))

    # Minibatch Gradient Descent with Momentum
    fitted_theta_2, cost_iter_2 = mini_batch_grad_desc(x_train, y_train, theta, momentum=0.9)
    acc2 = accuracy(fitted_theta_2, x_test, y_test)
    print("Minibatch Gradient Descent with Momentum Accuracy: {}".format(acc2))

    # Minibatch Gradient Descent with Adagrad
    fitted_theta_3, cost_iter_3 = mini_batch_grad_desc(x_train, y_train, theta, adagrad=True)
    acc3 = accuracy(fitted_theta_3, x_test, y_test)
    print("Minibatch Gradient Descent with Adagrad Accuracy: {}".format(acc3))


    # Plot Testing
    #plot_line(x_test, y_test, theta=fitted_theta)
    # Draw Cost Function loss
    plt.plot(range(len(cost_iter_1[0:])), cost_iter_1[0:], color='red', label='Mini Batch SGD')
    plt.plot(range(len(cost_iter_2[0:])), cost_iter_2[0:], color='yellow', label='Mini Batch SGD with Momentum')
    plt.plot(range(len(cost_iter_3[0:])), cost_iter_3[0:], color='green',
              label='Mini Batch SDG with Adagrad')
    plt.legend(bbox_to_anchor=(1., 1.), loc=0, borderaxespad=0.)
    plt.show()