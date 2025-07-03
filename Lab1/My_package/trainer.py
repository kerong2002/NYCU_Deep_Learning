'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

# 進度條函式庫
from tqdm import tqdm

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

    def train(self, X_train, y_train, epochs, log_interval=1000):
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
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            # 1. 前向傳播，得到預測值
            predictions = self.model.forward(X_train)
            
            # 2. 計算損失
            loss = self.loss_fn.loss(predictions, y_train)
            loss_history.append(loss)
            
            # 3. 計算損失的梯度
            grad = self.loss_fn.grad(predictions, y_train)
            
            # 4. 反向傳播，計算各層的梯度
            self.model.backward(grad)
            
            # 5. 使用優化器更新權重
            self.optimizer.step(self.model)

            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
                
        return loss_history