# Spring 2025, 535507 Deep Learning
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

# Lab5: Value-based RL – Task 1 (CartPole-v1 Vanilla DQN)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    """
    權重初始化函數。
    對卷積層(Conv2d)和線性層(Linear)使用 Kaiming He 初始化，
    這有助於ReLU激活函數的網路穩定訓練。
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    深度 Q 網路 (Deep Q-Network)。
    輸入為狀態，輸出為每個動作的Q值。
    """
    def __init__(self, num_actions, input_dim=4):
        super(DQN, self).__init__()
        # CartPole state = (x, v, θ, ω)
        # 網路結構：多層感知機 (MLP)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        )

    def forward(self, x):
        """定義前向傳播：將狀態 x 傳入網路，得到 Q-value。"""
        return self.network(x)


class AtariPreprocessor:
    """
        用於 Atari ,CartPole 用不到。
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)],
                            maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    """Task 1 不用 PER"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity

    # 以下留空
    def add(self, transition, error):  pass
    def sample(self, batch_size):      pass
    def update_priorities(self, indices, errors):  pass


class DQNAgent:
    """
    DQN 代理人類別。
    負責與環境互動、管理經驗回放緩衝區、以及訓練神經網路。
    """
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.policy_network = DQN(self.num_actions).to(self.device)
        self.policy_network.apply(init_weights)
        self.target_network = DQN(self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=args.lr)
        self.replay_buffer = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.train_count = 0 # 總訓練步數
        self.env_steps = 0   # 總環境互動步數
        self.best_eval_reward = 0  # CartPole 的初始最佳獎勵設為 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.student_id = args.student_id
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state_tensor):
        """
        使用 Epsilon-Greedy 策略來選擇動作。
        """
        # 如果小於 epsilon，隨機探索
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # 否則，利用 policy network 選擇最佳動作
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return q_values.argmax().item()

    def train_agent(self, num_episodes=1000):
        """主訓練迴圈"""
        evaluation_rewards_history = deque(maxlen=20) # 用於計算最近20個評估回合的平均獎勵
        
        for episode_index in range(num_episodes):
            state, info = self.env.reset() # state shape: (4,); action size: 2
            done = False
            total_training_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
                action = self.select_action(state_tensor)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.append((state, action, reward, next_state, done))

                if len(self.replay_buffer) > self.replay_start_size:
                    for _ in range(self.train_per_step):
                        self.update_model()

                state = next_state
                total_training_reward += reward
                step_count += 1
                self.env_steps += 1
            
            # --- 每回合都進行評估，並計算最近20回合的移動平均 ---
            current_eval_reward = self.evaluate_performance()
            evaluation_rewards_history.append(current_eval_reward)
            avg_eval_reward = np.mean(evaluation_rewards_history)

            # 每個回合結束時記錄訓練獎勵和評估的移動平均獎勵
            wandb.log({
                "Reward/Training Episode Reward": total_training_reward,
                "Eval/Average Evaluation Reward": avg_eval_reward
            }, step=self.env_steps)
            print(f"Episode {episode_index}: Training Reward: {total_training_reward}, Avg Eval Reward (last {len(evaluation_rewards_history)}): {avg_eval_reward:.2f}, Epsilon: {self.epsilon:.4f}, Env Steps: {self.env_steps}")

            # 每 20 個回合再進行一次日誌記錄和模型儲存的檢查
            if (episode_index + 1) % 20 == 0:
                wandb.log({
                    "Eval/Current Evaluation Reward": current_eval_reward
                }, step=self.env_steps)

                if avg_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = avg_eval_reward
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task1.pt")
                    torch.save(self.policy_network.state_dict(), model_path)
                    print(f"*** New best model saved to {model_path} with avg eval reward {self.best_eval_reward:.2f} ***")

            # 提早停止條件：最近20個回合的平均評估獎勵超過490
            if len(evaluation_rewards_history) >= 20 and avg_eval_reward > 490:
                print(f"Solved! Average evaluation reward over last 20 episodes is {avg_eval_reward:.2f}. Stopping training.")
                # 確保在退出前儲存最終模型
                if avg_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = avg_eval_reward
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task1.pt")
                    torch.save(self.policy_network.state_dict(), model_path)
                    print(f"*** Final best model saved to {model_path} with avg reward {self.best_eval_reward:.2f} ***")
                return # 直接退出 run 函數

    def evaluate_performance(self):
        """
        在測試環境中評估代理人的表現，回傳單個回合的總獎勵。
        """
        state, info = self.test_env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.policy_network(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        return total_reward
    
    def update_model(self):
        """
        從經驗回放緩衝區中取樣並進行一次模型更新。
        """
        if len(self.replay_buffer) < self.replay_start_size:
            return
        
        # Epsilon-greedy 探索的衰減
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        self.policy_network.train() # 設定為訓練模式
        self.target_network.eval() # 設定為評估模式
        
        # 從回放緩衝區中採樣一個 mini-batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # 將採樣的數據轉換為 PyTorch 張量
        state_batch = torch.from_numpy(np.array(state_batch)).float().to(self.device)
        next_state_batch = torch.from_numpy(np.array(next_state_batch)).float().to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).view(-1, 1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).view(-1, 1).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.bool).view(-1, 1).to(self.device)

        # 計算當前 Q 值 (Q(s,a))
        current_q_values = self.policy_network(state_batch).gather(1, action_batch)

        # 計算目標 Q 值 (r + gamma * max(Q_target(s',a')))
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].view(-1, 1)
            target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)
        
        # 計算損失 (MSE Loss)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 優化器步驟：清零梯度、反向傳播、更新權重
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        # 定期更新目標網路的權重
        if self.train_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # 每 1000 次訓練步數記錄損失
        if self.train_count % 1000 == 0:
            wandb.log({"Train/Loss": loss.item()}, step=self.env_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN for CartPole-v1 - Final Stable Version")
    
    # --- 基本設定 ---
    parser.add_argument("--env-name", type=str, default="CartPole-v1", help="Gym 環境名稱")
    parser.add_argument("--student-id", type=str, default="314551010", help="學號")
    parser.add_argument("--save-dir", type=str, default=".", help="模型儲存目錄")
    parser.add_argument("--wandb-run-name", type=str, default="T1_314551010", help="WandB 執行名稱")
    parser.add_argument("--wandb-group-name", type=str, default="Lab5_314551010", help="WandB 群組名稱")
    parser.add_argument("--gpu", type=str, default="0", help="要使用的 GPU 編號")
    
    # --- 學習相關超參數 ---
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率 (Adam 優化器)")
    parser.add_argument("--batch-size", type=int, default=128, help="每次訓練使用的樣本數")
    parser.add_argument("--memory-size", type=int, default=100000, help="經驗回放緩衝區的最大容量")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="折扣因子 (gamma)")
    
    # --- Epsilon-Greedy 策略相關超參數 ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon 起始值")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon 乘法衰減率")
    parser.add_argument("--epsilon-min", type=float, default=0.001, help="Epsilon 最小值")
    
    # --- 訓練流程相關超參數 ---
    parser.add_argument("--target-update-frequency", type=int, default=500, help="目標網路更新頻率 (以訓練步數計)")
    parser.add_argument("--replay-start-size", type=int, default=256, help="開始訓練前, buffer 至少要有的經驗數量")
    parser.add_argument("--train-per-step", type=int, default=1, help="每與環境互動一步，訓練網路的次數")
    
    # --- 執行回合與步數限制 ---
    parser.add_argument("--num-episodes", type=int, default=1000, help="總共要執行的回合數")
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="每個回合最大步數 (設大一點避免提前結束)")
    
    args = parser.parse_args()

    # 初始化 Weights & Biases (WandB) 用於實驗追蹤
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, group=args.wandb_group_name, config=vars(args))
    
    # 建立並執行 Agent
    agent = DQNAgent(args=args)
    agent.train_agent(num_episodes=args.num_episodes)
    
    # 結束 WandB 執行
    wandb.finish()
