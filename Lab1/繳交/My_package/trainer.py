'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

import numpy as np

def calculate_accuracy(predictions, labels):
    """計算準確率"""
    # 將預測值轉換為 0 或 1
    predicted_labels = (predictions > 0.5).astype(int)
    # 計算相同的數量
    correct_predictions = np.sum(predicted_labels == labels)
    # 計算準確率
    accuracy = correct_predictions / len(labels)
    return accuracy

class Trainer:
    """
    訓練器類別，負責執行模型的訓練迴圈。
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        初始化訓練器。
        
        參數:
            model: 要訓練的模型物件。
            loss_fn: 使用的損失函數物件。
            optimizer: 使用的優化器物件。
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs, log_interval=5000):
        """
        執行訓練迴圈。
        
        參數:
            X_train (np.array): 訓練資料。
            y_train (np.array): 訓練標籤。
            epochs (int): 訓練週期數。
            log_interval (int): 輸出日誌的間隔週期數。
            
        返回:
            list: 每個週期的損失值歷史紀錄。
        """
        loss_history = []
        acc_history = []
        for epoch in range(epochs):
            # 1. 前向傳播，得到預測值
            predictions = self.model.forward(X_train)
            
            # 2. 計算損失和準確率
            loss = self.loss_fn.loss(predictions, y_train)
            accuracy = calculate_accuracy(predictions, y_train)
            loss_history.append(loss)
            acc_history.append(accuracy)
            
            # 3. 計算損失的梯度
            grad = self.loss_fn.grad(predictions, y_train)
            
            # 4. 反向傳播，計算各層的梯度
            self.model.backward(grad)
            
            # 5. 使用優化器更新權重
            self.optimizer.step(self.model)

            if (epoch + 1) % log_interval == 0:
                print(f"epoch {epoch+1} loss : {loss}")
                
        return loss_history, acc_history