# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)  # 训练数据点数目
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 2 * x + 5 + np.random.randn(m)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(2)
# w = np.zeros(2)

alpha = 0.01  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数
finish = 0  # 终止标志
error_list = []
batch_size = 10;
shuffle_data = 1;

while count < loop_max:
    count += 1

    #   sum_m = np.zeros(2)
    for batch in range(0, m, batch_size):
        grad = np.dot((np.dot(input_data[batch: batch + batch_size, :], w.T)
                       - target_data[batch: batch + batch_size]).T, input_data[batch: batch + batch_size, :])
        w = w - alpha * (grad * 1.0 / batch_size)  # 注意步长alpha的取值,过大会导致振荡

        error_list.append(np.sum(grad) ** 2)
    # 判断是否已收敛
        if np.linalg.norm(w - error) < epsilon:
            finish = 1
            break
        else:
            error = w
        print 'loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1])

# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print 'intercept = %s slope = %s' % (intercept, slope)

plt.plot(range(len(error_list[0:100])), error_list[0:100])
plt.show()

plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()