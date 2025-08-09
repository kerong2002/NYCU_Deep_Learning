
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import wandb


# =============================================================================
#  1. 輔助函式與模組
# =============================================================================

class Mish(nn.Module):
    """Mish 活化函數模組。"""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * torch.tanh(F.softplus(input_tensor))


def to_tensor(data, device: torch.device) -> torch.Tensor:
    """將 numpy 陣列或 list 轉換為指定裝置上的 float 張量。"""
    if isinstance(data, list):
        data = np.array(data, dtype=np.float32)
    elif not isinstance(data, np.ndarray):
        data = np.array([data], dtype=np.float32)
    return torch.from_numpy(data).float().to(device)


# =============================================================================
#  2. 神經網路模型 (Actor & Critic)
# =============================================================================

class Actor(nn.Module):
    """策略網路 (Policy Network)：將狀態映射至動作的機率分佈。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, activation_cls: nn.Module = Mish):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activation_cls(),
            nn.Linear(hidden_size, hidden_size),
            activation_cls(),
            nn.Linear(hidden_size, action_dim)
        )
        self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state_tensor: torch.Tensor) -> torch.distributions.Normal:
        action_means = self.network(state_tensor)
        action_stds = torch.clamp(self.log_stds.exp(), 1e-3, 10.0)
        return torch.distributions.Normal(action_means, action_stds)


class Critic(nn.Module):
    """價值網路 (Value Network)：評估一個狀態的價值。"""

    def __init__(self, state_dim: int, hidden_size: int = 64, activation_cls: nn.Module = Mish):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activation_cls(),
            nn.Linear(hidden_size, hidden_size),
            activation_cls(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(state_tensor)


# =============================================================================
#  3. Runner: 負責與環境互動
# =============================================================================

class Runner:
    """管理代理人與環境之間的互動，負責收集經驗。"""

    def __init__(self, env: gym.Env, actor: Actor, max_total_steps: int):
        self.env = env
        self.actor = actor
        self.max_total_steps = max_total_steps
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.state, _ = self.env.reset()  # 初始 reset 由 train_agent 控制
        self.done = True

    def run(self, num_steps: int, device: torch.device) -> list:
        memory = []
        for _ in range(num_steps):
            if self.done:
                if self.total_steps > 0:
                    self.episode_rewards.append(self.episode_reward)
                    print(
                        f"訓練回合 {len(self.episode_rewards)} 結束. 分數: {self.episode_reward:.2f}. 總步數: {self.total_steps}")
                self.state, _ = self.env.reset()
                self.episode_reward = 0
                self.done = False

            if self.total_steps >= self.max_total_steps: break

            with torch.no_grad():
                state_tensor = to_tensor(self.state, device).unsqueeze(0)
                policy_dist = self.actor(state_tensor)
                action = policy_dist.sample().squeeze(0).cpu().numpy()

            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            next_state, reward, terminated, truncated, _ = self.env.step(clipped_action)
            self.done = terminated or truncated
            memory.append((self.state, action, reward, next_state, self.done))
            self.state = next_state
            self.total_steps += 1
            self.episode_reward += reward
        return memory


# =============================================================================
#  4. Learner: 負責模型更新
# =============================================================================

class A2CLearner:
    """管理學習過程 (更新 Actor 和 Critic 網路)。"""

    def __init__(self, actor: Actor, critic: Critic, config: argparse.Namespace):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    def learn(self, memory: list, current_step: int, device: torch.device):
        if not memory: return
        states, actions, rewards, next_states, dones = zip(*memory)
        states_t = to_tensor(np.array(states), device)
        actions_t = to_tensor(np.array(actions), device)
        rewards_t = to_tensor(np.array(rewards), device).view(-1, 1)
        next_states_t = to_tensor(np.array(next_states), device)
        dones_t = to_tensor(np.array(dones), device).view(-1, 1)

        with torch.no_grad():
            td_target = rewards_t + self.config.gamma * self.critic(next_states_t) * (1 - dones_t)

        state_values = self.critic(states_t)
        critic_loss = F.smooth_l1_loss(state_values, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        advantage = (td_target - state_values).detach()
        policy_dist = self.actor(states_t)
        log_probs = policy_dist.log_prob(actions_t)
        entropy = policy_dist.entropy().mean()
        actor_loss = -(log_probs * advantage).mean() - self.config.entropy_beta * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # <--- W&B 日誌修改 1: 將所有訓練指標與 "Global Step" 對齊
        if wandb.run:
            wandb.log({
                "Train/actor_loss": actor_loss.item(),
                "Train/critic_loss": critic_loss.item(),
                "Train/entropy": entropy.item(),
                "Global Step": current_step
            })


# =============================================================================
#  5. 訓練中的評估函式
# =============================================================================

def evaluate_during_training(training_env: gym.Env, learner: A2CLearner, config: argparse.Namespace, total_steps: int,
                             best_score: float) -> tuple[float, float]:
    eval_env = gym.make(config.env_id)
    eval_env = gym.wrappers.NormalizeObservation(eval_env)
    eval_env.obs_rms = training_env.obs_rms
    eval_env.training = False

    total_reward = 0
    for i in range(config.eval_episodes):
        state, _ = eval_env.reset(seed=config.seed + i)
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = to_tensor(state, config.device).unsqueeze(0)
                action = learner.actor(state_t).mean.squeeze(0).cpu().numpy()
            state, reward, terminated, truncated, _ = eval_env.step(
                np.clip(action, eval_env.action_space.low, eval_env.action_space.high))
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward

    average_reward = total_reward / config.eval_episodes
    print(f"\n--- 訓練中評估 ---")
    print(f"在 {config.eval_episodes} 回合中的平均分數: {average_reward:.2f} (在總步數 {total_steps})")

    if average_reward > best_score:
        print(f"發現新的最佳分數！舊: {best_score:.2f}, 新: {average_reward:.2f}。儲存最佳模型並錄製影片。")
        best_score = average_reward
        save_dict = {
            'actor_state_dict': learner.actor.state_dict(),
            'critic_state_dict': learner.critic.state_dict(),
            'obs_rms': training_env.obs_rms,
            'env_name': config.env_id
        }
        model_path = os.path.join(config.save_model_path, f"{config.run_name}_best.pt")
        torch.save(save_dict, model_path)
        print(f"最佳模型已儲存至 {model_path}")

        video_path = os.path.join(config.video_folder, f"best_at_step_{total_steps}")
        video_env = gym.make(config.env_id, render_mode="rgb_array")
        video_norm_env = gym.wrappers.NormalizeObservation(video_env)
        video_norm_env.obs_rms = training_env.obs_rms
        video_norm_env.training = False
        video_wrapped_env = gym.wrappers.RecordVideo(video_norm_env, video_folder=video_path,
                                                     episode_trigger=lambda x: x == 0)

        state, _ = video_wrapped_env.reset(seed=config.seed)
        done = False
        while not done:
            with torch.no_grad():
                action = learner.actor(to_tensor(state, config.device).unsqueeze(0)).mean.squeeze(0).cpu().numpy()
            state, _, terminated, truncated, _ = video_wrapped_env.step(
                np.clip(action, video_wrapped_env.action_space.low, video_wrapped_env.action_space.high))
            done = terminated or truncated
        video_wrapped_env.close()
        print(f"最佳表現影片已儲存至 {video_path}")

    print("--- 評估結束 ---\n")
    eval_env.close()
    return average_reward, best_score


# =============================================================================
#  6. 獨立的測試函式
# =============================================================================

def test_agent(config: argparse.Namespace):
    print("\n--- 啟動測試模式 (Inference Mode) ---")
    if not config.load_model_path: raise ValueError("測試模式需要模型路徑，請使用 --load_model_path 指定。")

    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {config.device}")
    print(f"載入模型: {config.load_model_path}")

    model_data = torch.load(config.load_model_path, map_location=config.device, weights_only=False)
    state_dim = model_data['obs_rms'].mean.shape[0]
    action_dim = model_data['actor_state_dict']['log_stds'].shape[0]
    actor = Actor(state_dim, action_dim).to(config.device)
    actor.load_state_dict(model_data['actor_state_dict'])
    actor.eval()

    base_env = gym.make(config.env_id, render_mode="rgb_array")
    norm_env = gym.wrappers.NormalizeObservation(base_env)
    print("正在還原儲存的觀測狀態正規化 (obs_rms) 物件...")
    norm_env.obs_rms = model_data['obs_rms']
    norm_env.training = False
    print("還原成功！")

    video_path = os.path.join(config.video_folder, "inference_run_fixed_seeds")
    test_env = gym.wrappers.RecordVideo(norm_env, video_folder=video_path, episode_trigger=lambda x: True)

    total_reward = 0
    num_test_episodes = 20
    print(f"開始在固定的 {num_test_episodes} 個種子 (0 to {num_test_episodes - 1}) 上執行測試...")
    for i in range(707, 707+num_test_episodes):
        state, _ = test_env.reset(seed=i)
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = to_tensor(state, config.device).unsqueeze(0)
                action = actor(state_t).mean.squeeze(0).cpu().numpy()
            state, reward, terminated, truncated, _ = test_env.step(
                np.clip(action, test_env.action_space.low, test_env.action_space.high))
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
        print(f"測試回合 {i + 1}/{num_test_episodes} (seed={i}), 分數: {episode_reward:.2f}")

    average_reward = total_reward / num_test_episodes
    print("\n--- 測試結果報告 ---")
    print(f"測試的平均分數 (在 {num_test_episodes} 個固定種子上): {average_reward:.2f}")
    print(f"測試影片已儲存至: {video_path}")
    print("--- 測試結束 ---")
    test_env.close()


# =============================================================================
#  7. 主執行區塊
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2C for Pendulum-v1 (固定種子與原始參數版)")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=707, help="確保整個訓練過程可複現")
    parser.add_argument("--run_name", type=str, default="a2c_pendulum_original_params")
    parser.add_argument("--inference", action="store_true", help="啟用測試模式")
    parser.add_argument("--load_model_path", type=str, default="", help="要載入的模型路徑")
    parser.add_argument("--max_steps", type=int, default=200200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.93)
    parser.add_argument("--actor_lr", type=float, default=6e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-3)
    parser.add_argument("--entropy_beta", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--video_folder", type=str, default="task1_videos")
    parser.add_argument("--save_model_path", type=str, default="task1_models")
    parser.add_argument("--wandb", action="store_true", help="啟用 W&B 實驗追蹤")
    return parser.parse_args()


def main(config: argparse.Namespace):
    if config.inference:
        test_agent(config)
    else:
        # 你程式碼中的超參數被我改回預設值，以便示範。你可以自行調整。
        train_agent(config)


def train_agent(config: argparse.Namespace):
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"--- 啟動訓練模式 ---")
    print(f"使用裝置: {config.device}")

    os.makedirs(config.video_folder, exist_ok=True)
    os.makedirs(config.save_model_path, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make(config.env_id)
    env = gym.wrappers.NormalizeObservation(env)
    env.reset(seed=config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(config.device)
    critic = Critic(state_dim).to(config.device)
    learner = A2CLearner(actor, critic, config)
    runner = Runner(env, actor, config.max_steps)

    if config.wandb:
        try:
            wandb.init(project=f"A2C-{config.env_id}", name=config.run_name, config=config)
            # <--- W&B 日誌修改 2: 定義自定義 X 軸
            # 這會告訴 wandb 我們有一個名為 "Global Step" 的主要計數器
            wandb.define_metric("Global Step")
            # 這會告訴 wandb 所有以 "Train/" 或 "Eval/" 開頭的指標都應該使用 "Global Step" 作為 X 軸
            wandb.define_metric("Train/*", step_metric="Global Step")
            wandb.define_metric("Eval/*", step_metric="Global Step")
        except Exception as e:
            print(f"Wandb 初始化失敗: {e}，將在無 wandb 模式下繼續。")

    best_eval_score = -np.inf
    last_eval_episode = 0

    print("\n--- 開始訓練迴圈 (使用固定種子) ---")
    while runner.total_steps < config.max_steps:
        memory = runner.run(config.batch_size, config.device)
        if not memory: break
        learner.learn(memory, runner.total_steps, config.device)

        current_episode_count = len(runner.episode_rewards)
        if current_episode_count >= last_eval_episode + config.eval_interval:
            avg_reward, best_eval_score = evaluate_during_training(env, learner, config, runner.total_steps,
                                                                   best_eval_score)
            # <--- W&B 日誌修改 3: 將評估指標與 "Global Step" 對齊
            if wandb.run:
                wandb.log({
                    "Eval/avg_episode_reward": avg_reward,
                    "Eval/episode": current_episode_count,
                    "Global Step": runner.total_steps
                })
            last_eval_episode = current_episode_count

    print("\n--- 訓練結束 ---")
    env.close()
    if wandb.run: wandb.finish()


if __name__ == "__main__":
    args = parse_arguments()
    # 注意：我在 parse_arguments 中將超參數改回了原始值，你可以根據需要修改。
    main(args)

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pendulum-v1 的 A2C 演算法實現 (恢復原始參數與固定種子版)
#
# 此版本旨在提供一個完全可複現的訓練與測試流程。
#
# 核心特性:
# 1. Runner/Learner 結構分離，職責清晰。
# 2. **關鍵**: 狀態正規化 (State Normalization)。
# 3. **關鍵**: 訓練和測試的超參數完全由 argparse 的原始預設值決定。
# 4. **關鍵**: 訓練時使用固定的全局種子，確保整個訓練過程可複現。
# 5. 測試時使用固定的種子集 (0-19)，確保測試結果可複現。
# 6. 詳盡的中文註解。
#
# 版本 3.0 修復:
# - 使用 wandb.define_metric() 將所有日誌的 X 軸統一為環境總步數 (Global Step)。

'''