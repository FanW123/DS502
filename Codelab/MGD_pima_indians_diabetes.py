from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import sklearn
import sklearn.metrics as metrics



print('\nRescale using sklearn')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
array = data_frame.values


# Separate array into input and output components
X = array[:, 0:8]
y = array[:, 8]

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Summarize transformed data
set_printoptions(precision=3)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

loop_max = 10000
epsilon = 1e-3


np.random.seed(0)
w = np.random.randn(9)
# w = np.zeros(2)

alpha = 0.001
diff = 0.
error = np.zeros(9)
count = 0
finish = 0
error_list = []
batch_size = 10
shuffle_data = 1

while count < loop_max:
    count += 1
    m = len(x_train)
    #   sum_m = np.zeros(2)
    for batch in range(0,m, batch_size):
        grad = np.dot((np.dot(x_train[batch: batch + batch_size, :], w.T)
                       - y_train[batch: batch + batch_size]).T, x_train[batch: batch + batch_size, :])
        w = w - alpha * (grad * 1.0 / batch_size)

        error_list.append(np.sum(grad) ** 2)

        if np.linalg.norm(w - error) < epsilon:
            finish = 1
            break
        else:
            error = w

cls = linear_model.LogisticRegression()

cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)

print cm
print "F1 score: %f" % sklearn.metrics.f1_score(y_test, y_pred)