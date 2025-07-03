'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

class Sequential:
    """
    一個簡單的神經網路模型容器，可依序堆疊多個神經網路層。
    提供 forward（前向傳播）、backward（反向傳播）、predict（預測）等常用功能。
    """

    def __init__(self, layers=None):
        """
        初始化循序模型。

        參數:
            layers (list, optional): 一個神經網路層的列表，依序堆疊。
        """
        # 如果使用者有提供 layers，就使用該列表；否則初始化為空列表
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """
        向模型中新增一個神經網路層。

        參數:
            layer: 要加入的層（必須具備 forward 和 backward 方法）。
        """
        self.layers.append(layer)

    def forward(self, inputs):
        """
        執行整個模型的前向傳播。

        參數:
            inputs (np.array): 輸入資料，形狀通常為 (樣本數, 特徵數)。

        返回:
            np.array: 最終輸出結果。
        """
        output = inputs  # 初始輸入為 X
        for layer in self.layers:
            # 將輸出依序傳給下一層
            output = layer.forward(output)
        return output

    def backward(self, grad):
        """
        執行整個模型的反向傳播，用來計算每層參數的梯度。

        參數:
            grad (np.array): 從損失函數傳入的梯度（通常是 dL/dy）。
        """
        # 依序從最後一層向前一層傳遞誤差
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X):
        """
        使用訓練好的模型來對資料進行預測（二元分類）。

        參數:
            X (np.array): 輸入資料。

        返回:
            np.array: 二元預測結果，0 或 1。
        """
        # 進行一次前向傳播，得到每筆資料的預測機率（例如 sigmoid 的輸出）
        y_pred_proba = self.forward(X)
        # 如果大於 0.5 則判斷為 1，否則為 0
        return (y_pred_proba > 0.5).astype(int)
