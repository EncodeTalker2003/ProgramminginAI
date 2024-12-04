"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
# from task0_autodiff import *
# from task0_operators import *
import numpy as np
import torch
from torchvision import datasets, transforms

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    ## 请于此填写你的代码
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_set = datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    X_tr = train_set.data.numpy().astype(np.float64)
    X_tr = X_tr.reshape((X_tr.shape[0], -1))
    # print("shape and dtype are", X_tr.shape, X_tr.dtype, X_tr[0])
    X_tr -= np.min(X_tr)
    X_tr /= np.max(X_tr)
    y_tr = train_set.targets.numpy()
    
    X_te = test_set.data.numpy().astype(np.float64)
    X_te = X_te.reshape((X_te.shape[0], -1))
    X_te -= np.min(X_te)
    X_te /= np.max(X_te)
    y_te = test_set.targets.numpy()

    print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    return X_tr, y_tr, X_te, y_te

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """

    W1 = np.random.randn(n, hidden_dim).astype(np.float64) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float64) / np.sqrt(k)
    return [W1, W2]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    ## 请于此填写你的代码
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ## 请于此填写你的代码
    return (np.sum(np.log(np.sum(np.exp(Z), axis=1))) - np.sum(Z[np.arange(y.size), y]))/y.size

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ## 请于此填写你的代码
    total_iter = (y.size + batch - 1) // batch
    w1, w2 = weights[0], weights[1]
    for iter in range(total_iter):
        iter_l, iter_r = iter * batch, min((iter + 1) * batch, y.size)
        cur_x = X[iter_l:iter_r, :]
        cur_y = y[iter_l:iter_r]
        z1 = cur_x @ w1
        z1[z1 < 0] = 0
        g2 = np.exp(z1 @ w2)
        g2 = g2 / np.sum(g2, axis=1, keepdims=True)
        tmp = np.zeros((batch, cur_y.max() + 1))
        tmp[np.arange(batch), cur_y] = 1
        g2 = g2 - tmp
        g1 = np.zeros_like(z1)
        g1[z1 > 0] = 1
        g1 = g1 * (g2 @ w2.T)
        
        g1 = (cur_x.T @ g1) / batch
        g2 = (z1.T @ g2) / batch
        w1 -= lr * g1
        w2 -= lr * g2
    
    weights = [w1, w2]

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    ## 请于此填写你的代码
    total_iter = (y.size + batch - 1) // batch
    w1, w2 = weights[0], weights[1]
    epsilon = 1e-5
    m1, m2 = np.zeros_like(w1), np.zeros_like(w2)
    v1, v2 = np.zeros_like(w1), np.zeros_like(w2)
    for iter in range(total_iter):
        if iter>2:
            break
        iter_l, iter_r = iter * batch, min((iter + 1) * batch, y.size)
        cur_x = X[iter_l:iter_r, :]
        cur_y = y[iter_l:iter_r]
        z1 = cur_x @ w1
        z1[z1 < 0] = 0
        g2 = np.exp(z1 @ w2)
        g2 = g2 / np.sum(g2, axis=1, keepdims=True)
        tmp = np.zeros((batch, cur_y.max() + 1))
        tmp[np.arange(batch), cur_y] = 1
        g2 = g2 - tmp
        g1 = np.zeros_like(z1)
        g1[z1 > 0] = 1
        g1 = g1 * (g2 @ w2.T)
        
        g1 = (cur_x.T @ g1) / batch
        g2 = (z1.T @ g2) / batch
        
        m1 = beta1 * m1 + (1 - beta1) * g1
        m2 = beta1 * m2 + (1 - beta1) * g2
        v1 = beta2 * v1 + (1 - beta2) * g1 * g1
        v2 = beta2 * v2 + (1 - beta2) * g2 * g2
        m1_hat = m1 / (1 - beta1 ** (iter + 1))
        m2_hat = m2 / (1 - beta1 ** (iter + 1))
        v1_hat = v1 / (1 - beta2 ** (iter + 1))
        v2_hat = v2 / (1 - beta2 ** (iter + 1))
        w1 -= lr * m1_hat / (np.sqrt(v1_hat) + epsilon)
        w2 -= lr * m2_hat / (np.sqrt(v2_hat) + epsilon)
        # print(w1, w2)
    weights = [w1, w2]


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)
    

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(X_tr, weights), y_tr)
        test_loss, test_err = loss_err(forward(X_te, weights), y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    print("load data done")
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.2, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.2, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
    