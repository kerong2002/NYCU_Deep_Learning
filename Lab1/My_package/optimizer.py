'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import numpy as np

class Optimizer:
    """優化器的基礎類別"""
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, model):
        """更新模型的權重"""
        raise NotImplementedError

class SGD(Optimizer):
    """
    隨機梯度下降 (Stochastic Gradient Descent) 優化器。
    可選配動量 (Momentum)。
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
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