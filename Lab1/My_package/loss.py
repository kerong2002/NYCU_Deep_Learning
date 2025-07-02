'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import numpy as np

class Loss:
    """損失函數的基礎類別"""
    def loss(self, predicted, actual):
        """計算損失"""
        raise NotImplementedError

    def grad(self, predicted, actual):
        """計算損失函數對於預測輸出的梯度"""
        raise NotImplementedError

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

class BCELoss(Loss):
    """
    二元交叉熵 (Binary Cross-Entropy) 損失函數。
    專為二元分類問題設計。
    """
    def loss(self, predicted, actual):
        """
        計算二元交叉熵損失。
        為了避免 log(0) 的問題，會對預測值進行裁切。
        """
        n_samples = actual.shape[0]
        epsilon = 1e-12
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.sum(actual * np.log(predicted_clipped) + (1 - actual) * np.log(1 - predicted_clipped)) / n_samples

    def grad(self, predicted, actual):
        """
        計算二元交叉熵損失的梯度。
        """
        n_samples = actual.shape[0]
        epsilon = 1e-12
        predicted_clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return - (actual / predicted_clipped - (1 - actual) / (1 - predicted_clipped)) / n_samples