# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL – Task 2 (Atari Pong)
# 修正版本：整合參考範例，並修復 wandb 紀錄問題

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import os
from collections import deque
import wandb
import argparse
import ale_py  # 確保 Atari 環境被註冊


def init_weights(m):
    """
    權重初始化函數。
    對卷積層(Conv2d)和線性層(Linear)使用 Kaiming He 初始化，
    有助於使用ReLU激活函數的網路穩定訓練。
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    深度 Q 網路 (Deep Q-Network)。
    改為使用更深、更有效的 VGG-style 網路架構。
    Input shape: (batch, 4, 84, 84)
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
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

            nn.Flatten(),  # -> (N, 64 * 10 * 10 = 6400)

            # 全連接層
            nn.Linear(64 * 10 * 10, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.network.apply(init_weights)

    def forward(self, x):
        """
        定義前向傳播。
        注意：輸入的 x 應為 float32 型別, 範圍在 [0,1]
        """
        return self.network(x / 255.0)


class AtariPreprocessor:
    """
    Atari 遊戲畫面預處理器。
    - 將畫面轉為灰階
    - 縮放至 84x84
    - 堆疊多個 frame (frame stacking)
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """預處理單一畫面：灰階 -> 縮放"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # 返回 uint8 型別, (84, 84)

    def reset(self, obs):
        """重置環境時，用第一個畫面填滿 frame buffer"""
        frame = self.preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        # 回傳堆疊後的 frames, shape: (4, 84, 84), dtype: uint8
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        """執行一步時，將新畫面加入並回傳堆疊後結果"""
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class DQNAgent:
    """DQN 代理人"""

    def __init__(self, env_name="ALE/Pong-v5", args=None):
        # 初始化環境
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        # 設定計算裝置
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化 Q-Network 和 Target-Network
        self.policy_network = DQN(self.num_actions).to(self.device)
        self.target_network = DQN(self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        # 使用 Adam 優化器，eps 參數可增加訓練穩定性
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=args.lr, eps=1.5e-4)
        # 新增：學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)

        # 使用 deque 作為經驗回放緩衝區
        self.replay_buffer = deque(maxlen=args.memory_size)

        # 訓練相關參數
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min

        # 計數器
        self.train_count = 0  # 總訓練次數
        self.env_step_count = 0  # 環境互動總步數

        # 評估與儲存相關
        self.best_eval_reward = -21.0  # Pong 的初始最低分
        # 評估與儲存相關
        self.best_eval_reward = -21.0  # Pong 的初始最低分
        # 用於儲存最近評估獎勵的歷史記錄
        self.eval_rewards_history = deque(maxlen=20)

        # 訓練流程控制
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

        # 儲存路徑
        self.save_dir = args.save_dir
        self.student_id = args.student_id
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state_tensor):
        """使用 epsilon-greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            self.policy_network.eval()  # 設為評估模式
            q_values = self.policy_network(state_tensor)
        return q_values.argmax().item()

    def train_agent(self, num_episodes=4000):
        """主訓練迴圈"""
        for episode_index in range(num_episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_training_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
                action = self.select_action(state_tensor)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                self.replay_buffer.append((state, action, reward, next_state, float(done)))

                if len(self.replay_buffer) > self.replay_start_size:
                    for _ in range(self.train_per_step):
                        self.update_model()

                    # Epsilon 線性衰減
                    if self.epsilon > self.epsilon_min:
                        self.epsilon -= (args.epsilon_start - args.epsilon_min) / self.epsilon_decay_steps

                state = next_state
                total_training_reward += reward
                step_count += 1
                self.env_step_count += 1

            # --- 每回合結束後 ---
            print(
                f"Episode {episode_index:4d} | Training Reward: {total_training_reward:5.1f} | Steps: {self.env_step_count:7d} | ε: {self.epsilon:.4f}")

            # 修改：採用高效的「定期評估」策略
            # 每 20 個回合評估一次模型
            if (episode_index + 1) % 20 == 0:
                # 進行 20 個回合的評估
                current_eval_reward = self.evaluate_performance(episodes=20)
                self.eval_rewards_history.append(current_eval_reward)
                avg_eval_reward = np.mean(self.eval_rewards_history)

                # 更新學習率調度器
                self.scheduler.step(avg_eval_reward)

                print(f"\n--- Evaluation at Episode {episode_index + 1} ---")
                print(f"Avg Eval Reward (last {len(self.eval_rewards_history)} evals): {avg_eval_reward:.2f}")

                wandb.log({
                    "Evaluate Reward": current_eval_reward,
                    "Progress/Learning Rate": self.optimizer.param_groups[0]['lr']
                }, step=self.env_step_count)

                # 如果平均獎勵創新高，儲存模型
                if avg_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = avg_eval_reward
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task2.pt")
                    torch.save(self.policy_network.state_dict(), model_path)
                    print(f"*** New best model saved to {model_path} with avg reward {avg_eval_reward:.2f} ***\n")

                # 提早停止條件
                if len(self.eval_rewards_history) >= 20 and avg_eval_reward >= 19.5:
                    print(f"\n--- Solved! Average reward over last 20 evaluations is {avg_eval_reward:.2f}. ---")
                    print("Stopping training.")
                    # 確保在退出前儲存最終模型
                    if avg_eval_reward > self.best_eval_reward:
                        self.best_eval_reward = avg_eval_reward
                        model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task2.pt")
                        torch.save(self.policy_network.state_dict(), model_path)
                        print(
                            f"*** Final best model saved to {model_path} with avg reward {self.best_eval_reward:.2f} ***")
                    return  # 直接退出 run 函數

    def evaluate_performance(self, episodes=20):
        """評估模型表現，回傳數個回合的平均獎勵"""
        rewards = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done, total_reward = False, 0
            while not done:
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    self.policy_network.eval()
                    action = self.policy_network(state_tensor).argmax().item()

                next_obs, r, term, trunc, _ = self.test_env.step(action)
                done = term or trunc
                total_reward += r
                state = self.preprocessor.step(next_obs)
            rewards.append(total_reward)
        return np.mean(rewards)

    def update_model(self):
        """從 Replay Buffer 中取樣並進行一次模型更新"""
        self.train_count += 1
        self.policy_network.train()  # 設為訓練模式

        # 從 replay_buffer 中隨機取樣一個 batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # 將資料轉為 PyTorch Tensors
        state_batch = torch.from_numpy(np.stack(state_batch)).to(self.device)
        next_state_batch = torch.from_numpy(np.stack(next_state_batch)).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        # --- 計算 Q-value 和 Target Q-value ---
        # 1. 計算當前 state 的 Q-value
        current_q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 2. 計算 target Q-value
        with torch.no_grad():
            # 使用 target network 估計下一個 state 的 Q-value
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            # 計算 TD target
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # 修改：使用 Smooth L1 Loss
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # --- 更新網路 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        # --- 定期更新 Target Network ---
        if self.train_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        # --- 定期紀錄訓練資訊 ---
        if self.train_count % 1000 == 0:
            wandb.log({
                "Train/Loss": loss.item(),
                "Progress/Epsilon": self.epsilon,
            }, step=self.env_step_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN for Atari Pong")

    # --- 實驗設定 ---
    parser.add_argument("--student-id", type=str, default="314551010", help="學號")
    parser.add_argument("--save-dir", type=str, default=".", help="模型儲存目錄")
    parser.add_argument("--wandb-run-name", type=str, default="T2_314551010", help="WandB 執行名稱")
    parser.add_argument("--wandb-group-name", type=str, default="Lab5_314551010", help="WandB 群組名稱")
    parser.add_argument("--gpu", type=str, default="0", help="要使用的 GPU 編號")
    parser.add_argument("--num-episodes", type=int, default=5000, help="總共要執行的回合數")

    # --- DQN 超參數 ---
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--memory-size", type=int, default=300_000, help="經驗回放緩衝區大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="折扣因子 gamma")

    # --- Epsilon-greedy 策略參數 ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon 起始值")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Epsilon 最小值")
    parser.add_argument("--epsilon-decay-steps", type=int, default=1_000_000,
                        help="Epsilon 從 start 衰減到 min 所需的總步數")

    # --- 訓練流程控制 ---
    parser.add_argument("--target-update-frequency", type=int, default=4000, help="Target network 更新頻率 (訓練次數)")
    parser.add_argument("--replay-start-size", type=int, default=20_000, help="開始訓練前, buffer 至少要有的經驗數量")
    parser.add_argument("--max-episode-steps", type=int, default=10_000, help="每個回合最大步數")
    parser.add_argument("--train-per-step", type=int, default=1, help="每與環境互動一步, 訓練幾次網路")

    args = parser.parse_args()

    # --- 初始化 WandB ---
    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, group=args.wandb_group_name, config=vars(args))

    # --- 建立並執行 Agent ---
    agent = DQNAgent(args=args)
    agent.train_agent(num_episodes=args.num_episodes)

    wandb.finish()