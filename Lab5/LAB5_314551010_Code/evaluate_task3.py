# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL – Task 3 (Evaluation Script)
# Date: 2025/07/30

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import os
from collections import deque
import argparse
import ale_py  # 確保 Atari 環境被註冊


# =================================================================================
# Model and Preprocessor (必須與 dqn_task3.py 中的定義完全一致)
# =================================================================================

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration (雜訊線性層).
    參考論文: "Noisy Networks for Exploration" (Fortunato et al., 2018)
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        # 評估時，只使用平均值 mu，以得到確定的輸出
        return nn.functional.linear(x, self.weight_mu, self.bias_mu)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, NoisyLinear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """DQN 網路架構 (Dueling an Noisy) (與 dqn_task3.py 同步)。"""
    def __init__(self, num_actions, noisy=False):
        super(DQN, self).__init__()
        self.noisy = noisy
        Linear = NoisyLinear if noisy else nn.Linear
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        feature_output_dim = 64 * 7 * 7
        self.value_stream = nn.Sequential(
            Linear(feature_output_dim, 512), nn.ReLU(),
            Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            Linear(feature_output_dim, 512), nn.ReLU(),
            Linear(512, num_actions)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = x / 255.0
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class AtariPreprocessor:
    """Atari 遊戲畫面預處理器 (與 dqn_task3.py 同步)"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = gray[34:194, :]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        _, thresholded = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return thresholded

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# =================================================================================
# 評估主程式
# =================================================================================

def evaluate_model(model_path, noisy_net, device, num_episodes=20):
    """
    載入並評估指定的 DQN 模型。

    Args:
        model_path (str): 儲存的模型權重檔案路徑 (.pt)。
        noisy_net (bool): 模型是否使用 Noisy Networks。
        device (torch.device): 計算裝置 (CPU or CUDA)。
        num_episodes (int): 要執行評估的回合數。
    """
    # 1. 初始化環境和預處理器
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    # 2. 載入模型
    model = DQN(num_actions, noisy=noisy_net).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 '{model_path}'。請檢查路徑是否正確。")
        return
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return

    model.eval()  # 設定為評估模式

    total_rewards = []

    # 從 model_path 中解析環境步數
    try:
        env_steps = os.path.basename(model_path).split('_')[-1].split('.')[0]
    except:
        env_steps = "N/A"

    print(f"--- 開始評估模型: {os.path.basename(model_path)} ---")

    # 3. 執行評估迴圈 (使用固定的種子)
    for i in range(num_episodes):
        seed = i  # 使用 0 到 19 的種子
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)

        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).to(device).unsqueeze(0).float()
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = preprocessor.step(next_obs)

        total_rewards.append(episode_reward)
        print(f"Environment steps: {env_steps}, seed: {seed:2d}, eval reward: {episode_reward}")

    # 4. 計算並顯示平均分數
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward: {avg_reward:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評估 Task 3 的 DQN 模型")
    parser.add_argument("--model_path", type=str, required=True, help="要評估的模型權重檔案路徑 (.pt)")
    parser.add_argument("--noisy", action='store_true', help="如果模型是使用 Noisy Networks 訓練的，請加上此旗標")
    parser.add_argument("--gpu", type=str, default="0", help="要使用的 GPU 編號")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    evaluate_model(args.model_path, args.noisy, device)