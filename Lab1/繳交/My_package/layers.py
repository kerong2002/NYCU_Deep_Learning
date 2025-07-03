'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import numpy as np

class Layer:
    """
    神經網路層的基礎類別（抽象類別）。
    所有自訂層都應繼承此類，並實作 forward() 與 backward() 方法。
    """
    def __init__(self):
        self.params = {}  # 儲存此層的參數，如 W, b
        self.grads = {}   # 儲存參數對 Loss 的梯度 ∂L/∂param

    def forward(self, inputs):
        """前向傳播（需在子類中實作）"""
        raise NotImplementedError

    def backward(self, grad):
        """反向傳播（需在子類中實作）"""
        raise NotImplementedError

class Linear(Layer):
    """
    線性層（全連接層），執行線性轉換：y = xW + b
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 初始化參數：W 為 (輸入維度, 輸出維度)，使用小隨機值；b 為 (輸出維度,)
        self.params['W'] = np.random.randn(input_dim, output_dim) * 0.1
        self.params['b'] = np.zeros(output_dim)
        self.inputs = None  # 用於反向傳播計算 ∂L/∂W

    def forward(self, inputs):
        """
        前向傳播：輸入乘上權重，加上偏置。
        """
        self.inputs = inputs  # 儲存輸入以備 backward 使用
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, grad):
        """
        反向傳播：
        - 計算 ∂L/∂W, ∂L/∂b 並儲存在 grads 中
        - 回傳對輸入的梯度 ∂L/∂x（給前一層使用）
        """
        self.grads['b'] = np.sum(grad, axis=0)  # 對偏置求梯度（沿著 batch 維度加總）
        self.grads['W'] = np.dot(self.inputs.T, grad)  # 對權重求梯度
        return np.dot(grad, self.params['W'].T)  # 回傳 ∂L/∂x，傳給前一層

class Sigmoid(Layer):
    """
    Sigmoid 活化函數層，輸出範圍在 (0, 1) 之間。
    適用於二元分類輸出層。
    """

    def __init__(self):
        super().__init__()
        self.output = None  # 儲存前向輸出值，供 backward 使用

    def forward(self, inputs):
        """
        前向傳播：應用 sigmoid 函數。
        """
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad):
        """
        反向傳播：
        使用 sigmoid 的導數：σ'(x) = σ(x)(1 - σ(x))
        再乘上來自下一層的梯度（鏈式法則）
        """
        return grad * (self.output * (1 - self.output))

class ReLU(Layer):
    """
    ReLU（Rectified Linear Unit）：
    若 x > 0 則傳遞 x，否則為 0。
    適合解決梯度消失問題。
    """

    def __init__(self):
        super().__init__()
        self.inputs = None  # 儲存輸入值以備反向使用

    def forward(self, inputs):
        """
        前向傳播：ReLU(x) = max(0, x)
        """
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad):
        """
        反向傳播：
        ReLU 對輸入的導數為：
        - 1 if x > 0
        - 0 if x <= 0
        所以只保留對應 x > 0 的梯度
        """
        grad_copy = grad.copy()
        grad_copy[self.inputs <= 0] = 0
        return grad_copy