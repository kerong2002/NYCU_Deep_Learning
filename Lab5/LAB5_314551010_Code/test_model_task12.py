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

# --- 可重複使用的類別 (從訓練腳本中改編) ---

def init_weights(m):
    """權重初始化"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    將 Q-value 分解為 State-Value 和 Action-Advantage 兩部分。
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        # 共享的卷積層，用於特徵提取
        # 共享的卷積層 (更深的 VGG-style)，用於特徵提取
        self.feature_extractor = nn.Sequential(
            # Block 1: 4x84x84 -> 32x42x42
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x42x42 -> 64x21x21
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64x21x21 -> 64x10x10
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(), # -> (N, 64 * 10 * 10 = 6400)
        )

        # State-Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 10 * 10, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Action-Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 10 * 10, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Dueling DQN forward pass does not normalize here if input is already float
        if x.dtype == torch.uint8:
            x = x / 255.0 # 正規化
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # 合併 V(s) 和 A(s,a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DQN(nn.Module):
    """
    通用 Q-Network。
    此版本修正了 CNN 架構以匹配 dqn_Task2.py 中的模型。
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape

        if len(self.input_shape) == 1: # 1D 輸入 (e.g., CartPole for Task 1)
            # 建立 MLP
            self.network = nn.Sequential(
                nn.Linear(input_shape[0], 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_actions)
            )
        elif len(self.input_shape) == 3: # 3D 輸入 (e.g., Atari frames for Task 2)
            # 建立 CNN (VGG-style architecture)
            # 這個架構必須與 dqn_Task2.py 中的架構完全相同
            self.network = nn.Sequential(
                # Block 1: 4x84x84 -> 32x42x42
                nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Block 2: 32x42x42 -> 64x21x21
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Block 3: 64x21x21 -> 64x10x10
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten(), # -> (N, 64 * 10 * 10 = 6400)

                # 全連接層
                nn.Linear(64 * 10 * 10, 512), nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            raise ValueError(f"不支援的輸入維度: {self.input_shape}")

        # 權重初始化 (在測試時非必要，但保持一致性)
        self.network.apply(init_weights)

    def forward(self, x):
        # 將 Atari 的 uint8 畫面正規化
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.network(x)

class AtariPreprocessor:
    """Atari 畫面預處理器"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames.extend([frame] * self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# --- 評估主函數 ---

def evaluate_task2(args):
    """專為 Task2 設計的評估函數，使用固定的種子"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --- Task2 環境設定 ---
    env_name = "ALE/Pong-v5"
    is_atari = True
    frame_stack = 4
    preprocessor = AtariPreprocessor(frame_stack=frame_stack)
    input_shape = (frame_stack, 84, 84)

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    # --- 載入模型 ---
    model = DQN(input_shape, env.action_space.n).to(device)
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
    
    num_episodes = 20 # 固定20回合
    print(f"開始為 Task2 評估，固定執行 {num_episodes} 回合...")

    # --- 評估迴圈 (固定種子) ---
    for ep in range(num_episodes):
        # 為每個 episode 設定固定種子
        obs, _ = env.reset(seed=ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []

        while not done:
            state_tensor = torch.from_numpy(state).to(torch.uint8).unsqueeze(0).to(device)

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
    """通用評估函數 (目前僅用於 Task1)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --- 根據任務設定環境 ---
    if args.task != "Task1":
        print(f"此評估函數僅適用於 Task1。若要評估 Task2，請確保主程式正確調用 evaluate_task2。")
        return

    env_name = "CartPole-v1"
    input_shape = (4,)
    is_atari = False
    preprocessor = None

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    # --- 載入模型 ---
    model = DQN(input_shape, env.action_space.n).to(device)
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
        state = obs  # CartPole 不需要預處理器
        done = False
        total_reward = 0
        frames = []

        # 修正評估迴圈，確保能錄到最後一幀
        while not done:
            state_tensor = torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            # 執行動作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # **重要修正**: 在執行動作 *之後* 進行 render，才能捕捉到該動作導致的畫面
            if render_mode == "rgb_array":
                frames.append(env.render())

            total_reward += reward
            state = next_obs

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}/{args.episodes} | Reward: {total_reward}")

        # 儲存影片
        if render_mode == "rgb_array" and frames:
            out_path = os.path.join(args.output_dir, f"eval_{args.task}_ep{ep+1}_reward{total_reward:.0f}.mp4")
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
    parser.add_argument("--task", type=str, choices=["Task1", "Task2"], required=True, help="指定要評估的任務 (Task1: CartPole, Task2: Pong)")
    parser.add_argument("--model-path", type=str, required=True, help="已訓練模型的 .pt 檔案路徑")
    parser.add_argument("--output-dir", type=str, default="./eval_videos", help="儲存評估影片的資料夾")
    parser.add_argument("--episodes", type=int, default=20, help="評估的回合數 (僅對 Task1 有效)")
    parser.add_argument("--render", action="store_true", help="顯示即時畫面 (human mode)，而不是儲存影片")
    args = parser.parse_args()

    if args.task == "Task2":
        try:
            import ale_py
            print("ale_py 已匯入")
        except ImportError:
            print("錯誤: 測試 Atari 模型需要 'ale-py'。請執行 'pip install ale-py'")
            exit()
        evaluate_task2(args)
    elif args.task == "Task1":
        evaluate(args)
