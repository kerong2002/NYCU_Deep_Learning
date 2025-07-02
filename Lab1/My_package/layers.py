'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import numpy as np

class Layer:
    """
    神經網路層的基礎類別。
    """
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        """前向傳播"""
        raise NotImplementedError

    def backward(self, grad):
        """反向傳播"""
        raise NotImplementedError

class Linear(Layer):
    """
    線性層 (全連接層)。
    執行 y = xW + b 的運算。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 初始化權重和偏置
        self.params['W'] = np.random.randn(input_dim, output_dim) * 0.1
        self.params['b'] = np.zeros(output_dim)
        self.inputs = None

    def forward(self, inputs):
        """
        執行前向傳播。
        """
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, grad):
        """
        執行反向傳播，計算權重和輸入的梯度。
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['W'] = np.dot(self.inputs.T, grad)
        return np.dot(grad, self.params['W'].T)

class Sigmoid(Layer):
    """
    Sigmoid 活化函數層。
    """
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs):
        """
        執行前向傳播。
        """
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad):
        """
        執行反向傳播。
        """
        return grad * (self.output * (1 - self.output))

class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) 活化函數層。
    """
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        """
        執行前向傳播。
        """
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad):
        """
        執行反向傳播。
        """
        grad_copy = grad.copy()
        grad_copy[self.inputs <= 0] = 0
        return grad_copy