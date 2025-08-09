#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip for Walker2d-v4
#
# v3.0 更新日誌 (參考成功範例進行優化):
# - [優化] 調整超參數以適應 Walker2d-v4 的高難度。
#   - 提高 gamma 至 0.99，讓 Agent 更重視長期回報。
#   - 降低 entropy_beta 至 0.01，減少不必要的隨機探索，穩定學習。
#   - 調整學習率至更穩定的範圍。
# - [優化] 將神經網路的隱藏層從 64 擴大到 256，增強模型容量。
# - [新增] 加入學習率線性衰減 (anneal_lr) 功能，幫助後期收斂。
# - 保留了 v2.1 的所有錯誤修復和日誌輸出格式。

import os
import random
import argparse
import time
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# 偵測 wandb 是否安裝
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
#  1. 神經網路模型 (Actor & Critic)
# =============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """對神經網路的層進行正交初始化 (Orthogonal Initialization)。"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """策略網路 (Policy Network)，負責根據當前狀態生成動作。"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # [優化] 增加網路容量以應對更複雜的任務
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        action_mean = self.net(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    """價值網路 (Value Network)，負責評估一個狀態的價值 (V-value)。"""

    def __init__(self, state_dim: int):
        super().__init__()
        # [優化] 增加網路容量
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# =============================================================================
#  2. PPO 核心演算法輔助函式
# =============================================================================

def compute_gae(
        next_value: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        device: torch.device
) -> torch.Tensor:
    """計算廣義優勢估計 (Generalized Advantage Estimation, GAE)。"""
    advantages = torch.zeros_like(rewards).to(device)
    last_gae_lam = 0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    return advantages


def ppo_iter(
        batch_size: int,
        minibatch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
):
    """一個迭代器，用於將一個大的 batch 數據隨機切分成多個 minibatch。"""
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_indices = indices[start:end]
        yield states[mb_indices], actions[mb_indices], log_probs[mb_indices], returns[mb_indices], advantages[
            mb_indices]


# =============================================================================
#  3. PPO Agent 主類別
# =============================================================================

class PPOAgent:
    """PPO Agent，整合了模型、互動和學習的完整邏輯。"""

    def __init__(self, state_dim: int, action_dim: int, args: argparse.Namespace):
        """初始化 Agent"""
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用裝置: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

    def select_action(self, state: np.ndarray, is_test: bool = False) -> np.ndarray:
        """從給定的狀態中選擇一個動作。"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action, dist = self.actor(state_tensor)
        if is_test:
            action = dist.mean
        return action.cpu().numpy().flatten()

    def update_model(self, states, actions, log_probs, returns, advantages) -> dict:
        """使用收集到的 rollout 數據更新模型。"""
        minibatch_size = self.args.batch_size // self.args.num_minibatches

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.args.n_epochs):
            for state, action, old_log_prob, return_, adv in ppo_iter(
                    batch_size=self.args.batch_size,
                    minibatch_size=minibatch_size,
                    states=states, actions=actions, log_probs=log_probs,
                    returns=returns, advantages=advantages,
            ):
                _, new_dist = self.actor(state)
                new_value = self.critic(state)
                new_log_prob = new_dist.log_prob(action).sum(1)
                entropy = new_dist.entropy().sum(1)

                v_loss = F.smooth_l1_loss(new_value.squeeze(), return_)
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                surr1 = adv * ratio
                surr2 = adv * torch.clamp(ratio, 1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef)
                pg_loss = -torch.min(surr1, surr2).mean()
                actor_loss = pg_loss - self.args.entropy_beta * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()

        return {"policy_loss": pg_loss.item(), "value_loss": v_loss.item()}


# =============================================================================
#  4. 訓練與評估流程
# =============================================================================

def evaluate_agent(eval_env: gym.Env, agent: PPOAgent, args: argparse.Namespace) -> float:
    """在訓練過程中評估 Agent 的表現。"""
    total_reward = 0
    for i in range(args.eval_episodes):
        state, _ = eval_env.reset(seed=args.seed + i)
        done = False
        while not done:
            action = agent.select_action(state, is_test=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / args.eval_episodes


def train_agent(args: argparse.Namespace):
    """主訓練函式"""
    print("--- 啟動 PPO 訓練模式 ---")

    writer = SummaryWriter(f"runs/{args.run_name}")
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project=f"PPO-{args.env_id}", name=args.run_name, config=vars(args))

    os.makedirs(args.save_model_path, exist_ok=True)

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, args)

    global_step = 0
    episode_count = 0
    best_eval_score = -np.inf
    last_eval_episode = 0
    state, _ = env.reset(seed=args.seed)
    start_time = time.time()

    recent_rewards = deque(maxlen=20)

    student_id = "314551010"
    checkpoint_steps = {
        1_000_000: "1m", 1_500_000: "1p5m", 2_000_000: "2m",
        2_500_000: "2p5m", 3_000_000: "3m",
    }
    sorted_checkpoint_steps = sorted(checkpoint_steps.items())
    saved_checkpoints = [False] * len(sorted_checkpoint_steps)

    print("\n--- 開始 PPO 訓練迴圈 ---")
    while global_step < args.max_steps:
        # [新增] 學習率線性衰減
        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.max_steps
            lr_now = frac * args.actor_lr
            agent.actor_optimizer.param_groups[0]["lr"] = lr_now
            lr_now = frac * args.critic_lr
            agent.critic_optimizer.param_groups[0]["lr"] = lr_now

        states_buf = torch.zeros((args.batch_size, state_dim)).to(agent.device)
        actions_buf = torch.zeros((args.batch_size, action_dim)).to(agent.device)
        log_probs_buf = torch.zeros((args.batch_size,)).to(agent.device)
        rewards_buf = torch.zeros((args.batch_size,)).to(agent.device)
        dones_buf = torch.zeros((args.batch_size,)).to(agent.device)
        values_buf = torch.zeros((args.batch_size,)).to(agent.device)

        for step in range(args.batch_size):
            global_step += 1
            states_buf[step] = torch.from_numpy(state)

            with torch.no_grad():
                action_tensor, dist = agent.actor(torch.from_numpy(state).float().to(agent.device).unsqueeze(0))
                action = action_tensor.cpu().numpy().flatten()
                log_prob = dist.log_prob(action_tensor).sum()
                value = agent.critic(torch.from_numpy(state).float().to(agent.device).unsqueeze(0))

            actions_buf[step] = action_tensor
            log_probs_buf[step] = log_prob
            values_buf[step] = value.flatten()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards_buf[step] = torch.tensor(reward).view(-1)
            dones_buf[step] = torch.tensor(float(done))
            state = next_state

            if done:
                state, _ = env.reset()

            if "episode" in info:
                episode_count += 1
                ep_return = info["episode"]["r"]
                recent_rewards.append(ep_return)

                print(f"回合 {episode_count} | 步數: {global_step} | 分數: {ep_return:.2f}")

                if episode_count % 20 == 0:
                    avg_reward_20 = np.mean(list(recent_rewards))
                    print(f"--- 最近 20 回合平均分數: {avg_reward_20:.2f} ---")
                    writer.add_scalar("charts/avg_20_ep_return", avg_reward_20, global_step)
                    if args.wandb and WANDB_AVAILABLE:
                        wandb.log({"charts/avg_20_ep_return": avg_reward_20}, step=global_step)

                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "charts/episodic_return": ep_return,
                        "charts/SPS": int(global_step / (time.time() - start_time))
                    }, step=global_step)

                if episode_count >= last_eval_episode + args.eval_interval_episodes:
                    last_eval_episode = episode_count

                    eval_env = gym.make(args.env_id)
                    eval_env = gym.wrappers.NormalizeObservation(eval_env)
                    eval_env.obs_rms = env.env.obs_rms
                    eval_env.training = False

                    avg_score = evaluate_agent(eval_env, agent, args)
                    writer.add_scalar("eval/avg_reward", avg_score, global_step)
                    if args.wandb and WANDB_AVAILABLE:
                        wandb.log({"eval/avg_reward": avg_score}, step=global_step)

                    print(f"\n[評估] Global Step: {global_step}, 平均分數: {avg_score:.2f}\n")

                    if avg_score > best_eval_score:
                        best_eval_score = avg_score
                        print(f"發現新的最佳分數！儲存模型...")
                        torch.save({
                            'actor_state_dict': agent.actor.state_dict(),
                            'critic_state_dict': agent.critic.state_dict(),
                            'obs_rms': env.env.obs_rms,
                        }, os.path.join(args.save_model_path, f"{args.run_name}_best.pt"))

                    eval_env.close()

        with torch.no_grad():
            next_value = agent.critic(torch.from_numpy(state).float().to(agent.device).unsqueeze(0)).reshape(1, -1)
        advantages = compute_gae(next_value, rewards_buf, dones_buf, values_buf, args.gamma, args.gae_lambda,
                                 agent.device)
        returns = advantages + values_buf

        log_data = agent.update_model(states_buf, actions_buf, log_probs_buf, returns, advantages)
        writer.add_scalar("losses/policy_loss", log_data["policy_loss"], global_step)
        writer.add_scalar("losses/value_loss", log_data["value_loss"], global_step)
        if args.wandb and WANDB_AVAILABLE:
            wandb.log(log_data, step=global_step)

        for i, (step_threshold, step_str) in enumerate(sorted_checkpoint_steps):
            if not saved_checkpoints[i] and global_step >= step_threshold:
                save_path = os.path.join(args.save_model_path, f"LAB7_{student_id}_task3_ppo_{step_str}.pt")
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'obs_rms': env.env.obs_rms,
                }, save_path)
                print(f"\n模型已於 {global_step} 步儲存至: {save_path}\n")
                saved_checkpoints[i] = True

    env.close()
    writer.close()
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


def test_agent(args: argparse.Namespace):
    """獨立的測試函式，載入模型並錄製影片。"""
    print("\n--- 啟動測試模式 (Inference Mode) ---")
    if not args.load_model_path:
        raise ValueError("測試模式需要模型路徑，請使用 --load_model_path 指定。")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 修正後程式碼
    checkpoint = torch.load(args.load_model_path, map_location=device, weights_only=False)

    obs_rms = checkpoint['obs_rms']
    state_dim = obs_rms.mean.shape[0]
    action_dim = checkpoint['actor_state_dict']['actor_logstd'].shape[1]

    agent = PPOAgent(state_dim, action_dim, args)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor.eval()
    agent.critic.eval()

    video_path = os.path.join(args.video_folder, f"inference_{args.run_name}")
    os.makedirs(video_path, exist_ok=True)
    test_env = gym.make(args.env_id, render_mode="rgb_array")
    test_env = gym.wrappers.RecordVideo(test_env, video_folder=video_path, episode_trigger=lambda x: True)
    test_env = gym.wrappers.ClipAction(test_env)
    test_env = gym.wrappers.NormalizeObservation(test_env)

    test_env.obs_rms = obs_rms
    test_env.training = False

    total_reward = 0
    num_test_episodes = 20
    print(f"開始在 {num_test_episodes} 個回合上執行測試...")
    for i in range(num_test_episodes):
        state, _ = test_env.reset(seed=args.seed + i)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, is_test=True)
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
        print(f"測試回合 {i + 1}/{num_test_episodes}, 分數: {episode_reward:.2f}")

    print(f"\n測試平均分數: {total_reward / num_test_episodes:.2f}")
    print(f"測試影片已儲存至: {video_path}")
    test_env.close()


# =============================================================================
#  5. 主執行區塊
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="PPO for Walker2d-v4")

    parser.add_argument("--env_id", type=str, default="Walker2d-v4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=777, help="確保整個訓練過程可複現")
    parser.add_argument("--run_name", type=str, default=f"ppo_walker_{int(time.time())}")
    parser.add_argument("--inference", action="store_true", help="啟用測試模式")
    parser.add_argument("--load_model_path", type=str, default="", help="要載入的模型路徑")
    parser.add_argument("--wandb", action="store_true", help="啟用 W&B 實驗追蹤")
    parser.add_argument("--video_folder", type=str, default="task3_videos")
    parser.add_argument("--save_model_path", type=str, default="task3_models")

    # --- [優化] 針對 Walker2d-v4 調整的訓練超參數 ---
    parser.add_argument("--max_steps", type=int, default=3_003_000)
    parser.add_argument("--batch_size", type=int, default=2000, help="每次策略更新收集的步數 (Rollout Buffer Size)")
    parser.add_argument("--actor_lr", type=float, default=5e-4, help="Actor 的學習率")
    parser.add_argument("--critic_lr", type=float, default=3e-3, help="Critic 的學習率")
    parser.add_argument("--anneal_lr", action="store_true", default=True, help="啟用學習率線性衰減")
    parser.add_argument("--n_epochs", type=int, default=20, help="每次更新時，重複學習數據的次數")
    parser.add_argument("--num_minibatches", type=int, default=32, help="將 batch 切分為多少個 minibatch")
    parser.add_argument("--gamma", type=float, default=0.92, help="折扣因子，Walker2d 需看得更遠")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 的 lambda 參數")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO 的裁剪係數 epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="價值函數損失的係數")
    parser.add_argument("--entropy_beta", type=float, default=0.02, help="熵係數，適度降低以穩定學習")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # --- 評估參數 ---
    parser.add_argument("--eval_interval_episodes", type=int, default=20, help="每隔多少回合進行一次評估")
    parser.add_argument("--eval_episodes", type=int, default=20, help="每次評估時運行的回合總數")
    # [新增] 為獨立測試新增參數
    parser.add_argument("--num_test_episodes", type=int, default=20, help="獨立測試時運行的回合總數")
    return parser.parse_args()


def main(config: argparse.Namespace):
    """主函式，根據 --inference 參數決定執行訓練或測試。"""
    seed_torch(config.seed)
    if config.inference:
        test_agent(config)
    else:
        train_agent(config)


def seed_torch(seed: int):
    """設定所有相關的隨機種子以確保可重現性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
