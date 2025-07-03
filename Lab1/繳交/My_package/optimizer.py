'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''


# Optimizer: GD, SGD, Adam, Adagrad


import numpy as np

class Optimizer:
    """優化器的基礎類別"""
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, model):
        """更新模型的權重"""
        raise NotImplementedError

# 批次梯度下降 (Batch Gradient Descent)
class GD(Optimizer):
    """
    批次梯度下降，使用整個訓練集的平均梯度更新參數。
    """
    def __init__(self, learning_rate=0.01):
        # 這一行會自動把 lr 存到 self.lr
        super().__init__(learning_rate)

    def step(self, model):
        for i, layer in enumerate(model.layers):
            # 尋找參數
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # 直接依梯度更新，無動量
                    layer.params[key] -= self.lr * layer.grads[key]

class SGD(Optimizer):
    """
    隨機梯度下降 (Stochastic Gradient Descent) 優化器。
    可選配動量 (Momentum)。
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        # 這一行會自動把 lr 存到 self.lr
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {} # 儲存每個參數的速度

    def step(self, model):
        """
        遍歷模型中的所有層，並更新其可訓練的參數。
        
        參數:
            model: 要優化的模型物件。
        """
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    param_key = f"layer_{i}_{key}" # 唯一的鍵來識別參數
                    
                    # 初始化速度
                    if param_key not in self.velocities:
                        self.velocities[param_key] = np.zeros_like(layer.params[key])
                    
                    # 計算速度 (velocity)
                    self.velocities[param_key] = (self.momentum * self.velocities[param_key] - 
                                                  self.lr * layer.grads[key])
                    
                    # 更新權重
                    layer.params[key] += self.velocities[param_key]

# Adam 優化器
class Adam(Optimizer):
    """
    Adaptive Moment Estimatio
    Adam 優化器，結合動量與 RMSProp，自動調整學習率。
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # 這一行會自動把 lr 存到 self.lr
        super().__init__(learning_rate)
        self.beta1 = beta1  # 動量衰減率
        self.beta2 = beta2  # 均方衰減率
        self.epsilon = epsilon
        self.m = {}  # 一階矩估計（動量）
        self.v = {}  # 二階矩估計（均方）
        self.t = 0   # 時間步長

    def step(self, model):
        self.t += 1
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    param_key = f"layer_{i}_{key}"
                    grad = layer.grads[key]

                    # 初始化 m, v
                    if param_key not in self.m:
                        self.m[param_key] = np.zeros_like(grad)
                        self.v[param_key] = np.zeros_like(grad)

                    # 更新一階矩估計（動量）
                    self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * grad
                    # 更新二階矩估計（均方）
                    self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (grad ** 2)

                    # 計算偏差修正
                    m_hat = self.m[param_key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[param_key] / (1 - self.beta2 ** self.t)

                    # 參數更新
                    update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    layer.params[key] -= update


class Adagrad(Optimizer):
    """
    Adaptive Gradient Algorithm
    Adagrad 優化器：自適應學習率，會根據歷史梯度調整每個參數的更新步伐。
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        # 這一行會自動把 lr 存到 self.lr
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.r = {}  # 儲存每個參數的累積平方梯度

    def step(self, model):
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    grad = layer.grads[key]
                    param_key = f"layer_{i}_{key}"

                    # 初始化累積梯度平方
                    if param_key not in self.r:
                        self.r[param_key] = np.zeros_like(grad)

                    # 累積平方梯度
                    self.r[param_key] += grad ** 2

                    # 更新參數（學習率會因 r_t 調整）
                    adjusted_lr = self.lr / (np.sqrt(self.r[param_key]) + self.epsilon)
                    layer.params[key] -= adjusted_lr * grad
