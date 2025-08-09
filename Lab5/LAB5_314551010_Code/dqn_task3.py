# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL – Task 3 (Advanced DQN)
# 修正版本: 整合 Double DQN, PER, Multi-step, Dueling DQN, Noisy Nets, 和更穩定的架構

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
import ale_py # 確保 Atari 環境被註冊

# --- 雜訊網路 (Noisy Networks) ---
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration (雜訊線性層).
    透過在網路的權重中加入可學習的雜訊，來驅動更有效的探索行為，
    取代傳統的 epsilon-greedy 策略。
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        # 可學習的權重和偏差的平均值 (mu) 和標準差 (sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        # 用於生成雜訊的非學習參數 (epsilon)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """初始化 mu 和 sigma 參數"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """生成新的雜訊樣本"""
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """前向傳播。訓練時使用帶有雜訊的權重，評估時只使用平均值。"""
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

def init_weights(m):
    """權重初始化 (跳過 NoisyLinear 層)"""
    if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, NoisyLinear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

# --- 網路架構 (Network Architecture) ---
class DQN(nn.Module):
    """
    Dueling DQN 網路架構。
    這個架構將Q值分解為兩個部分：狀態價值(V)和動作優勢(A)。
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    這樣做可以更穩定地學習狀態的價值，不受動作選擇的影響。
    """
    def __init__(self, num_actions, noisy=False):
        super(DQN, self).__init__()
        self.noisy = noisy
        # 根據是否使用Noisy Networks，選擇對應的線性層
        Linear = NoisyLinear if noisy else nn.Linear

        # 卷積層，用於從遊戲畫面中提取特徵
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # 計算卷積層輸出特徵的維度
        feature_output_dim = 64 * 7 * 7

        # 狀態價值流 (Value Stream)，估計V(s)
        self.value_stream = nn.Sequential(
            Linear(feature_output_dim, 512), nn.ReLU(),
            Linear(512, 1)
        )
        # 動作優勢流 (Advantage Stream)，估計A(s, a)
        self.advantage_stream = nn.Sequential(
            Linear(feature_output_dim, 512), nn.ReLU(),
            Linear(512, num_actions)
        )
        # 初始化網路權重
        self.apply(init_weights)

    def forward(self, x):
        """前向傳播，計算最終的 Q-value"""
        # 將輸入畫面像素值標準化到 [0, 1]
        x = x / 255.0
        # 提取特徵
        features = self.feature_extractor(x)
        # 計算價值和優勢
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # 根據 Dueling DQN 的公式合併價值和優勢
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        """重置所有 NoisyLinear 層的雜訊"""
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear): m.reset_noise()

# --- 畫面預處理 (Preprocessing) ---
class AtariPreprocessor:
    """Atari 遊戲畫面預處理器，負責將原始畫面轉換為神經網路的輸入格式。"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack  # 堆疊的幀數
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """預處理單一畫面：灰階 -> 裁切 -> 縮放 -> 二值化"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = gray[34:194, :] # 裁切掉分數等不重要區域
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        _, thresholded = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY) # 強調特徵
        return thresholded

    def reset(self, obs):
        """重置時用第一個畫面填滿 frame buffer"""
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        """將新畫面加入並回傳堆疊後結果"""
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# --- 經驗回放緩衝區 (Replay Buffer) ---
class PrioritizedReplayBuffer:
    """
    優先級經驗回放 (Prioritized Experience Replay, PER)。
    給予TD-error較大的經驗更高的抽樣優先級，讓學習更有效率。
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=1_000_000):
        self.capacity = capacity
        self.alpha = alpha  # 優先級強度
        self.beta = beta_start  # 重要性抽樣權重
        self.beta_increment = (1.0 - beta_start) / beta_steps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition):
        """新增經驗，給予最大優先級以確保新經驗能被學習"""
        if len(self.buffer) < self.capacity: self.buffer.append(transition)
        else: self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """根據優先級抽樣，並計算重要性抽樣(IS)權重以修正偏差"""
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, torch.from_numpy(weights).float()

    def update_priorities(self, indices, errors):
        """根據新的 TD-error 更新經驗的優先級"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (np.abs(error) + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, self.priorities.max())

    def __len__(self):
        return len(self.buffer)

# --- 代理人 (Agent) ---
class DQNAgent:
    """整合了多種先進技術的 DQN 代理人"""
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_network = DQN(self.num_actions, noisy=args.noisy).to(self.device)
        self.target_network = DQN(self.num_actions, noisy=args.noisy).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=args.lr, eps=1.5e-4)

        self.replay_buffer = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta_start=args.per_beta_start, beta_steps=args.total_steps)
        self.multi_step_buffer = deque(maxlen=args.multi_step)

        self.epsilon = args.epsilon_start
        self.train_count = 0
        self.env_step_count = 0
        self.best_eval_reward = -21.0

        self.save_dir = args.save_dir
        self.student_id = args.student_id
        os.makedirs(self.save_dir, exist_ok=True)

        self.checkpoint_steps = [400_000, 800_000, 1_200_000, 1_600_000, 2_000_000]
        self.next_checkpoint_idx = 0

    def select_action(self, state_tensor):
        """根據策略選擇動作"""
        if self.args.noisy:
            with torch.no_grad(): q_values = self.policy_network(state_tensor)
            return q_values.argmax().item()
        else:
            if random.random() < self.epsilon: return random.randint(0, self.num_actions - 1)
            with torch.no_grad(): q_values = self.policy_network(state_tensor)
            return q_values.argmax().item()

    def _get_multi_step_transition(self):
        """計算 N-step return"""
        reward = sum((self.args.discount_factor**i) * self.multi_step_buffer[i][2] for i in range(self.args.multi_step))
        state, action, _, _, _ = self.multi_step_buffer[0]
        _, _, _, next_state, done = self.multi_step_buffer[-1]
        return state, action, reward, next_state, done

    def train_agent(self):
        """主訓練迴圈"""
        episode_index = 0
        obs, _ = self.env.reset()
        state = self.preprocessor.reset(obs)

        while self.env_step_count < self.args.total_steps:
            if self.args.noisy:
                self.policy_network.reset_noise()
                self.target_network.reset_noise()

            done = False
            total_training_reward = 0

            while not done:
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0).float()
                action = self.select_action(state_tensor)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                total_training_reward += reward
                self.multi_step_buffer.append((state, action, reward, next_state, done))

                if len(self.multi_step_buffer) == self.args.multi_step:
                    self.replay_buffer.add(self._get_multi_step_transition())

                if len(self.replay_buffer) > self.args.replay_start_size and self.env_step_count % self.args.train_frequency == 0:
                    if not self.args.noisy:
                        fraction = min(1.0, self.env_step_count / self.args.epsilon_decay_steps)
                        self.epsilon = self.args.epsilon_start + fraction * (self.args.epsilon_min - self.args.epsilon_start)

                    for _ in range(self.args.train_per_step):
                        self.update_model()

                state = next_state
                self.env_step_count += 1

                if self.next_checkpoint_idx < len(self.checkpoint_steps) and self.env_step_count >= self.checkpoint_steps[self.next_checkpoint_idx]:
                    step = self.checkpoint_steps[self.next_checkpoint_idx]
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_{step}.pt")
                    torch.save(self.policy_network.state_dict(), model_path)
                    print(f"\n--- Saved checkpoint at {self.env_step_count} steps to {model_path} ---\n")
                    self.next_checkpoint_idx += 1

                if done: break

            print(f"Episode {episode_index:4d} | Reward: {total_training_reward:5.1f} | Steps: {self.env_step_count:7d} | ε: {self.epsilon:.4f}")

            if (episode_index + 1) % 20 == 0:
                eval_reward = self.evaluate_performance(episodes=20)
                if not self.args.no_wandb:
                    wandb.log({"Evaluate Reward": eval_reward}, step=self.env_step_count)
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_best.pt")
                    torch.save(self.policy_network.state_dict(), model_path)
                    print(f"*** New best model saved with avg reward {eval_reward:.2f} ***\n")

            episode_index += 1
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)

    def evaluate_performance(self, episodes=20):
        """評估模型表現"""
        rewards = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done, total_reward = False, 0
            while not done:
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0).float()
                with torch.no_grad():
                    self.policy_network.eval()
                    action = self.policy_network(state_tensor).argmax().item()
                next_obs, r, term, trunc, _ = self.test_env.step(action)
                done = term or trunc
                total_reward += r
                state = self.preprocessor.step(next_obs)
            rewards.append(total_reward)
        self.policy_network.train()
        avg_reward = np.mean(rewards)
        print(f"\n--- Evaluation Result: Avg Reward {avg_reward:.2f} over {episodes} episodes ---\n")
        return avg_reward

    def update_model(self):
        """從 Replay Buffer 中取樣並進行一次模型更新"""
        self.train_count += 1
        # 1. 從 Prioritized Replay Buffer 中抽樣一個批次的經驗
        samples, indices, weights = self.replay_buffer.sample(self.args.batch_size)

        # 2. 將經驗解構成各個部分
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*samples)

        # 3. 將數據轉換為 PyTorch 張量並移至指定裝置 (GPU/CPU)
        state_batch = torch.from_numpy(np.stack(state_batch)).to(self.device).float()
        next_state_batch = torch.from_numpy(np.stack(next_state_batch)).to(self.device).float()
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)

        # 4. 計算當前狀態-動作對的 Q 值 (Q(s,a))
        # 使用 policy_network 來計算，並用 gather 選出實際採取動作的 Q 值
        current_q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 5. 計算目標 Q 值 (Target Q-value) - Double DQN 的核心
        with torch.no_grad(): # 在這個區塊中，不計算梯度
            # a. 使用 policy_network 選擇下一個狀態的最佳動作 a'
            best_actions = self.policy_network(next_state_batch).argmax(1)
            # b. 使用 target_network 評估該動作 a' 的 Q 值
            next_q_values = self.target_network(next_state_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # c. 計算目標 Q 值: r + γ^n * Q_target(s', a') * (1 - done)
            target_q_values = reward_batch + (self.args.discount_factor ** self.args.multi_step) * next_q_values * (1 - done_batch)

        # 6. 更新 PER 的優先級
        # 計算 TD-error，並用它來更新 replay buffer 中樣本的優先級
        errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, errors)

        # 7. 計算損失函數
        # 使用 Smooth L1 Loss，並乘以重要性抽樣權重來修正偏差
        loss = (weights * nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

        # 8. 執行反向傳播和優化
        self.optimizer.zero_grad() # 清除舊梯度
        loss.backward() # 計算新梯度
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0) # 梯度裁剪，防止梯度爆炸
        self.optimizer.step() # 更新網路權重

        # 9. 定期更新 Target Network
        if self.train_count % self.args.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        # 10. 定期記錄訓練資訊到 WandB
        if self.train_count % 1000 == 0:
            if not self.args.no_wandb:
                wandb.log({"Train/Loss": loss.item(), "Progress/Epsilon": self.epsilon, "Progress/PER_Beta": self.replay_buffer.beta}, step=self.env_step_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced DQN for Atari Pong")

    parser.add_argument("--student-id", type=str, default="314551010", help="學號")
    parser.add_argument("--save-dir", type=str, default=".", help="模型儲存目錄")
    parser.add_argument("--wandb-run-name", type=str, default="T3_314551010", help="WandB 執行名稱")
    parser.add_argument("--wandb-group-name", type=str, default="Lab5_314551010", help="WandB 群組名稱")
    parser.add_argument("--gpu", type=str, default="0", help="要使用的 GPU 編號")
    parser.add_argument("--total-steps", type=int, default=2_100_000, help="總訓練步數")

    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--memory-size", type=int, default=100_000, help="經驗回放緩衝區大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="折扣因子 gamma")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon 起始值")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Epsilon 最小值")
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000, help="Epsilon 衰減總步數")

    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target network 更新頻率 (訓練次數)")
    parser.add_argument("--replay-start-size", type=int, default=10000, help="開始訓練前 buffer 經驗數量")
    parser.add_argument("--train-frequency", type=int, default=4, help="每多少環境互動步數，進行一次訓練週期")
    parser.add_argument("--train-per-step", type=int, default=1, help="每個訓練週期中，更新幾次網路")

    parser.add_argument("--multi-step", type=int, default=3, help="Multi-step return 的步數 n")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER 的 alpha 參數")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="PER 的 beta 起始值")
    parser.add_argument("--noisy", action='store_true', help="啟用 Noisy Networks for exploration")
    parser.add_argument("--no-wandb", action='store_true', help="停用 WandB 日誌紀錄")

    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(project="DLP-Lab5-DQN-Pong-Task3", name=args.wandb_run_name, group=args.wandb_group_name, config=vars(args))

    agent = DQNAgent(args=args)
    agent.train_agent()

    if not args.no_wandb:
        wandb.finish()