'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

class Sequential:
    """
    一個循序模型，可以將多個層堆疊在一起。
    """
    def __init__(self, layers=None):
        """
        初始化循序模型。
        
        參數:
            layers (list, optional): 一個包含神經網路層物件的列表。
        """
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """
        向模型中添加一個層。
        
        參數:
            layer: 要添加的層物件。
        """
        self.layers.append(layer)

    def forward(self, inputs):
        """
        執行完整的前向傳播。
        
        參數:
            inputs (np.array): 輸入資料。
        
        返回:
            np.array: 模型最終的輸出。
        """
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad):
        """
        執行完整的反向傳播。
        
        參數:
            grad (np.array): 來自損失函數的初始梯度。
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X):
        """
        使用訓練好的模型進行預測。
        
        參數:
            X (np.array): 輸入資料。
        
        返回:
            np.array: 預測的類別 (0 或 1)。
        """
        y_pred_proba = self.forward(X)
        return (y_pred_proba > 0.5).astype(int)