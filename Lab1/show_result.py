'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history):
    """
    繪製訓練損失曲線。
    
    參數:
        loss_history (list): 包含每個 epoch 損失值的列表。
    """
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def show_result(model, X, y_true):
    """
    視覺化比較真實標籤和預測結果，並計算準確率。
    
    參數:
        model: 訓練好的神經網路模型物件。
        X (np.array): 輸入資料。
        y_true (np.array): 真實標籤。
    """
    y_pred = model.predict(X)
    
    plt.figure(figsize=(12, 6))
    
    # 繪製真實標籤
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth", fontsize=16)
    plt.scatter(X[y_true.flatten() == 0][:, 0], X[y_true.flatten() == 0][:, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X[y_true.flatten() == 1][:, 0], X[y_true.flatten() == 1][:, 1], c='blue', marker='x', label='Class 1')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    # 繪製預測結果
    plt.subplot(1, 2, 2)
    plt.title("Prediction", fontsize=16)
    plt.scatter(X[y_pred.flatten() == 0][:, 0], X[y_pred.flatten() == 0][:, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X[y_pred.flatten() == 1][:, 0], X[y_pred.flatten() == 1][:, 1], c='blue', marker='x', label='Class 1')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 計算並顯示準確率
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"Accuracy: {accuracy:.2f}%")