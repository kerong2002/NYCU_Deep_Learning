# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL – 通用模型評估腳本 (Universal Model Evaluation Script)

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import os
from collections import deque
import argparse


# --- 可重複使用的類別 (從 dqn_task3.py 複製) ---

class NoisyLinear(nn.Module):
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
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                        self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
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

    def forward(self, x):
        x = x / 255.0
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class AtariPreprocessor:
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


# --- 評估主函數 ---

def evaluate_task3(args):
    """專為 Task3 設計的評估函數，使用固定的種子"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --- Task3 環境設定 ---
    env_name = "ALE/Pong-v5"
    preprocessor = AtariPreprocessor(frame_stack=4)
    
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    # --- 載入 Task3 模型 (DuelingDQN with Noisy Nets) ---
    model = DQN(env.action_space.n, noisy=True).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功從 {args.model_path} 載入 Task3 模型")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        env.close()
        return
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rewards = []
    
    num_episodes = 20 # 固定20回合
    print(f"開始為 Task3 評估，固定執行 {num_episodes} 回合...")

    # --- 評估迴圈 (固定種子) ---
    for ep in range(num_episodes):
        # 為每個 episode 設定固定種子
        obs, _ = env.reset(seed=ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render_mode == "rgb_array":
                frames.append(env.render())

            total_reward += reward
            state = preprocessor.step(next_obs)

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}/{num_episodes} (seed={ep}) | Reward: {total_reward}")

        if render_mode == "rgb_array" and frames:
            out_path = os.path.join(args.output_dir, f"eval_{args.task}_ep{ep+1}_seed{ep}_reward{total_reward:.0f}.mp4")
            imageio.mimsave(out_path, frames, fps=30)
            print(f"影片已儲存至: {out_path}")

    env.close()
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print("\n--- 評估總結 ---")
    print(f"任務: {args.task}")
    print(f"模型: {args.model_path}")
    print(f"平均獎勵 (共 {num_episodes} 回合, seeds 0-19): {avg_reward:.2f} +/- {std_reward:.2f}")
    print("------------------")


def evaluate(args):
    """通用評估函數 (用於 Task1 和 Task2)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --- 根據任務設定環境 ---
    is_atari = False
    preprocessor = None
    if args.task == "Task1":
        env_name = "CartPole-v1"
        input_shape = (4,)
    elif args.task == "Task2":
        env_name = "ALE/Pong-v5"
        is_atari = True
        frame_stack = 4
        preprocessor = AtariPreprocessor(frame_stack=frame_stack)
        input_shape = (frame_stack, 84, 84)
    else:
        raise ValueError(f"此函數不應處理 {args.task}")

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    # --- 載入模型 ---
    # 簡化版: 假設 Task1 和 Task2 使用不同的簡單模型
    if args.task == "Task1":
        model = nn.Sequential(nn.Linear(input_shape[0], 128), nn.ReLU(), nn.Linear(128, env.action_space.n)).to(device)
    else: # Task2
        class SimpleDQN(nn.Module):
            def __init__(self, input_shape, num_actions):
                super(SimpleDQN, self).__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
            def forward(self, x):
                return self.network(x / 255.0)
        model = SimpleDQN(input_shape, env.action_space.n).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功從 {args.model_path} 載入模型")
    except Exception as e:
        print(f"載入模型失敗: {e}")
        env.close()
        return
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rewards = []

    # --- 評估迴圈 ---
    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = preprocessor.reset(obs) if is_atari else obs
        done = False
        total_reward = 0
        frames = []

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if render_mode == "rgb_array":
                frames.append(env.render())
            total_reward += reward
            state = preprocessor.step(next_obs) if is_atari else next_obs

        all_rewards.append(total_reward)
        print(f"Episode {ep + 1}/{args.episodes} | Reward: {total_reward}")

        if render_mode == "rgb_array" and frames:
            out_path = os.path.join(args.output_dir, f"eval_{args.task}_ep{ep + 1}_reward{total_reward:.0f}.mp4")
            imageio.mimsave(out_path, frames, fps=30)
            print(f"影片已儲存至: {out_path}")

    env.close()
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print("\n--- 評估總結 ---")
    print(f"任務: {args.task}")
    print(f"模型: {args.model_path}")
    print(f"平均獎勵 (共 {args.episodes} 回合): {avg_reward:.2f} +/- {std_reward:.2f}")
    print("------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評估已訓練的 DQN 模型")
    parser.add_argument("--task", type=str, choices=["Task1", "Task2", "Task3"], required=True,
                        help="指定要評估的任務 (Task1: CartPole, Task2: Pong, Task3: Enhanced Pong)")
    parser.add_argument("--model-path", type=str, required=True, help="已訓練模型的 .pt 檔案路徑")
    parser.add_argument("--output-dir", type=str, default="./eval_videos", help="儲存評估影片的資料夾")
    parser.add_argument("--episodes", type=int, default=20, help="評估的回合數 (僅對 Task1/Task2 有效)")
    parser.add_argument("--render", action="store_true", help="顯示即時畫面 (human mode)，而不是儲存影片")
    args = parser.parse_args()

    # 確保 ale-py 已安裝 (對 Task2 和 Task3 都是必要的)
    if args.task in ["Task2", "Task3"]:
        try:
            import ale_py
            print("ale_py 已匯入")
        except ImportError:
            print("錯誤: 測試 Atari 模型需要 'ale-py'。請執行 'pip install ale-py'")
            exit()

    # 根據任務調用不同的評估函數
    if args.task == "Task3":
        evaluate_task3(args)
    else: # Task1 or Task2
        evaluate(args)
