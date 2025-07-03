'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

'''
Loss function
Mean Squared Error(mse)
CrossEntropyLoss(cross)
BCELoss(bce)
'''
import numpy as np

class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError("loss() 尚未實作")

    def grad(self, predicted, actual):
        raise NotImplementedError("grad() 尚未實作")

class MSE(Loss):
    """
    均方誤差 (Mean Squared Error) 損失函數。
    主要用於迴歸問題。
    """
    def loss(self, predicted, actual):
        """
        計算均方誤差。
        """
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted, actual):
        """
        計算均方誤差的梯度。
        """
        n_samples = actual.shape[0]
        return 2 * (predicted - actual) / n_samples

class CrossEntropyLoss(Loss):
    """
    多類別交叉熵損失函數 (Cross Entropy Loss)，
    用於 softmax 多類別分類問題。
    """
    def loss(self, predicted, actual, epsilon=1e-12):
        """
        計算交叉熵損失。
        - predicted: 預測機率分佈 (softmax 輸出)，shape: [n_classes, n_samples]
        - actual: one-hot 向量，shape: [n_classes, n_samples]
        """

        # 會把 array 中的每個元素壓在範圍 [min_value, max_value] 之內：
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return -np.mean(np.sum(actual * np.log(predicted), axis=0))  # 對每個樣本取交叉熵再平均

    def grad(self, predicted, actual, epsilon=1e-12):
        """
        交叉熵對 softmax 輸出的梯度。
        在 softmax + cross entropy 結合時，梯度為：
            grad = predicted - actual
        """
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return (predicted - actual) / predicted.shape[1]  # 除以樣本數做平均


class BCELoss(Loss):
    """
    二元交叉熵 (Binary Cross-Entropy) 損失函數。
    專為二元分類問題設計。
    """
    def loss(self, predicted, actual, epsilon=1e-12):
        """
        計算二元交叉熵損失。
        為了避免 log(0) 的問題，會對預測值進行裁切。
        """
        n_samples = actual.shape[0]
        # epsilon = 1e-12
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.sum(actual * np.log(predicted_clipped) + (1 - actual) * np.log(1 - predicted_clipped)) / n_samples

    def grad(self, predicted, actual, epsilon=1e-12):
        """
        計算二元交叉熵損失的梯度。
        """
        n_samples = actual.shape[0]
        # epsilon = 1e-12
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return - (actual / predicted_clipped - (1 - actual) / (1 - predicted_clipped)) / n_samples