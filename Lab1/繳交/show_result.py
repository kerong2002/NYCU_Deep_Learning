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

def plot_accuracy_curve(acc_history):
    """
    繪製訓練準確率曲線。
    
    參數:
        acc_history (list): 包含每個 epoch 準確率值的列表。
    """
    plt.figure()
    plt.plot(acc_history, color='green')
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def plot_loss_and_accuracy_curves(loss_history, acc_history):
    """
    在同一張圖上繪製訓練損失和準確率曲線（使用雙 Y 軸）。
    
    參數:
        loss_history (list): 包含每個 epoch 損失值的列表。
        acc_history (list): 包含每個 epoch 準確率值的列表。
    """
    fig, ax1 = plt.subplots()

    # 繪製 Loss 曲線
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(loss_history, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    # 創建共享 x 軸的第二個 y 軸
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(acc_history, color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 為了讓圖例(legend)能同時顯示兩條線的標籤
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Training Loss & Accuracy')
    fig.tight_layout()  # 自動調整邊距，防止標籤被裁切
    plt.show()

def show_result(model, X, y_true, loss_fn):
    """
    視覺化比較真實標籤和預測結果，並計算準確率。
    
    參數:
        model: 訓練好的神經網路模型物件。
        X (np.array): 輸入資料。
        y_true (np.array): 真實標籤。
        loss_fn: 損失函數物件。
    """
    predictions = model.forward(X)
    y_pred_labels = (predictions > 0.5).astype(int)
    
    loss = loss_fn.loss(predictions, y_true)
    accuracy = np.mean(y_pred_labels == y_true) * 100
    
    # --- 文字輸出 ---
    for i in range(len(X)):
        print(f"Iter{i+1} | Ground truth: {y_true[i][0]} | prediction: {predictions[i][0]:.5f} |")
    print(f"loss={loss:.5f} accuracy={accuracy:.2f}%")

    # --- 圖形化比較 ---
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
    plt.scatter(X[y_pred_labels.flatten() == 0][:, 0], X[y_pred_labels.flatten() == 0][:, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X[y_pred_labels.flatten() == 1][:, 0], X[y_pred_labels.flatten() == 1][:, 1], c='blue', marker='x', label='Class 1')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()