import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt('data.txt', delimiter=',')
x = points[:,0] # 第一列数据
y = points[:,1] # 第二列数据
# 用 scatter 画出散点图
plt.scatter(x, y)
plt.show()

#计算损失
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
    # 逐点计算【实际数据 yi 与 模型数据 f(xi) 的差值】的平方，然后求平均
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / M

#初始化模型超参数
alpha = 0.0001
initial_w = 0
initial_b = 0
batch_size = 10

#梯度下降算法
def grad_desc(points, initial_w, initial_b, alpha, batch_size):
    w = initial_w
    b = initial_b
    # 定义一个list保存所有的损失函数值，用来显示下降的过程
    cost_list = []

    for i in range(batch_size):
        # 先计算初始值的损失函数的值
        cost_list.append(compute_cost(w, b, points))
        w, b = step_grad_desc(w, b, alpha, points)

    return [w, b, cost_list]

def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)

    # 对每一个点带入公式求和
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += (current_w * x + current_b - y)

    # 用公式求当前梯度
    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b

    # 梯度下降，更新当前的 w 和 b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b

    return updated_w, updated_b
#测试梯度下降算法，计算最优解w和b
w, b, cost_list = grad_desc(points, initial_w, initial_b, alpha, batch_size)

print('w is:', w)
print('b is:', b)


cost = compute_cost(w, b, points)
print('cost is:', cost)

plt.plot(cost_list)
plt.show()

# 先用 scatter 画出2维散点图
plt.scatter(x, y)

# 针对每一个x，计算出预测的值
pred_y = w * x + b
# 再用 plot 画出2维直线图
plt.plot(x, pred_y, c='r')
plt.show()