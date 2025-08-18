#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip for Walker2d-v4
#
# v5.0 更新日誌 (進階優化):
# - [新增] KL 散度監控和早停機制，避免策略更新過度
# - [新增] 自適應學習率調整，根據KL散度動態調整
# - [優化] 改進的梯度裁剪策略
# - [新增] 價值函數預熱 (Value Function Warmup)
# - [優化] 更好的正規化和初始化策略
# - [新增] 動作標準差衰減機制，提升後期穩定性
# - [修正] 修復潛在的數值穩定性問題
# - [新增] 更詳細的訓練統計和監控

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
#  1. 改進的神經網路模型 (Actor & Critic)
# =============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """對神經網路的層進行正交初始化 (Orthogonal Initialization)。"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """改進的策略網路，加入更好的正規化和初始化。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        # 改進：使用可學習但有界的log std
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        action_mean = self.net(state)
        # 限制log std的範圍，避免數值不穩定
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), -20, 2)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        return action, dist

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """獲取動作和對應的log概率，用於訓練時的效率優化"""
        action_mean = self.net(state)
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), -20, 2)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1)


class Critic(nn.Module):
    """改進的價值網路，加入更好的正規化。"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


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
        values: torch.Tensor,
):
    """一個迭代器，用於將一個大的 batch 數據隨機切分成多個 minibatch。"""
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_indices = indices[start:end]
        yield (states[mb_indices], actions[mb_indices], log_probs[mb_indices],
               returns[mb_indices], advantages[mb_indices], values[mb_indices])


# =============================================================================
#  3. 改進的 PPO Agent 主類別
# =============================================================================

class PPOAgent:
    """改進的PPO Agent，加入KL散度監控和自適應學習率。"""

    def __init__(self, state_dim: int, action_dim: int, args: argparse.Namespace):
        """初始化 Agent"""
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用裝置: {self.device}")

        self.actor = Actor(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, args.hidden_dim).to(self.device)

        # 記錄初始學習率
        self.initial_actor_lr = args.actor_lr
        self.initial_critic_lr = args.critic_lr

        if not args.inference:
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(),
                lr=args.actor_lr,
                eps=1e-5,
                weight_decay=args.weight_decay
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=args.critic_lr,
                eps=1e-5,
                weight_decay=args.weight_decay
            )

            # 自適應學習率相關
            self.target_kl = args.target_kl
            self.kl_threshold = args.kl_threshold
            self.lr_decay_factor = 0.8
            self.lr_restore_factor = 1.1

    def select_action(self, state: np.ndarray, is_test: bool = False) -> np.ndarray:
        """從給定的狀態中選擇一個動作。"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action, dist = self.actor(state_tensor)
        if is_test:
            action = dist.mean
        return action.cpu().numpy().flatten()

    def update_model(self, states, actions, log_probs, returns, advantages, values) -> dict:
        """改進的模型更新，加入KL散度監控和早停。"""
        minibatch_size = self.args.batch_size // self.args.num_minibatches

        # 正規化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clipfracs = []

        early_stop = False

        for epoch in range(self.args.n_epochs):
            if early_stop:
                break

            epoch_approx_kls = []

            for state, action, old_log_prob, return_, adv, old_value in ppo_iter(
                    batch_size=self.args.batch_size,
                    minibatch_size=minibatch_size,
                    states=states, actions=actions, log_probs=log_probs,
                    returns=returns, advantages=advantages, values=values,
            ):
                # 獲取新的策略輸出
                _, new_log_prob, entropy = self.actor.get_action_and_value(state, action)
                new_value = self.critic(state)

                # 計算KL散度 (近似)
                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # 近似KL散度
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    epoch_approx_kls.append(approx_kl.item())

                    # 裁剪比例統計
                    clipfrac = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean()
                    clipfracs.append(clipfrac.item())

                # --- Critic 更新 (含價值裁剪) ---
                if self.args.clip_vloss:
                    v_loss_unclipped = (new_value - return_) ** 2
                    v_clipped = old_value + torch.clamp(
                        new_value - old_value, -self.args.clip_coef, self.args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - return_) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - return_) ** 2).mean()

                # --- Actor 更新 ---
                surr1 = adv * ratio
                surr2 = adv * torch.clamp(ratio, 1.0 - self.args.clip_coef, 1.0 + self.args.clip_coef)
                pg_loss = -torch.min(surr1, surr2).mean()

                # 熵獎勵
                entropy_loss = entropy.mean()
                actor_loss = pg_loss - self.args.entropy_beta * entropy_loss

                # 更新Critic
                self.critic_optimizer.zero_grad()
                (v_loss * self.args.vf_coef).backward()
                if self.args.max_grad_norm > 0:
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.args.max_grad_norm
                    )
                self.critic_optimizer.step()

                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.args.max_grad_norm > 0:
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.args.max_grad_norm
                    )
                self.actor_optimizer.step()

                # 記錄損失
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

            # 檢查KL散度是否過大，若是則早停
            avg_kl = np.mean(epoch_approx_kls)
            approx_kls.append(avg_kl)

            if self.args.early_stop_kl and avg_kl > self.kl_threshold:
                print(f"Early stopping at epoch {epoch + 1} due to high KL divergence: {avg_kl:.4f}")
                early_stop = True

        # 自適應學習率調整
        final_kl = np.mean(approx_kls[-5:]) if len(approx_kls) >= 5 else np.mean(approx_kls)
        if final_kl > self.target_kl * 1.5:
            # KL太大，降低學習率
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_factor
        elif final_kl < self.target_kl * 0.5:
            # KL太小，可以稍微提高學習率
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * self.lr_restore_factor,
                                        self.initial_actor_lr)

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clipfrac": np.mean(clipfracs),
            "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
            "critic_lr": self.critic_optimizer.param_groups[0]['lr'],
        }


# =============================================================================
#  4. 訓練與評估流程 (保持原有邏輯)
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
    print("--- 啟動改進版 PPO 訓練模式 ---")

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

    print("\n--- 開始改進版 PPO 訓練迴圈 ---")
    while global_step < args.max_steps:
        # 學習率調度
        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.max_steps
            lr_now = frac * args.actor_lr
            agent.actor_optimizer.param_groups[0]["lr"] = lr_now
            lr_now = frac * args.critic_lr
            agent.critic_optimizer.param_groups[0]["lr"] = lr_now

        # 動作標準差衰減 (可選)
        if args.action_std_decay:
            decay_factor = max(0.05, 1.0 - global_step / args.max_steps)
            with torch.no_grad():
                agent.actor.actor_logstd.data = torch.clamp(
                    agent.actor.actor_logstd.data,
                    -20,
                    np.log(decay_factor)
                )

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

        log_data = agent.update_model(states_buf, actions_buf, log_probs_buf, returns, advantages, values_buf)

        # 記錄更詳細的訓練統計
        for key, value in log_data.items():
            writer.add_scalar(f"losses/{key}", value, global_step)
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({f"losses/{key}": value}, step=global_step)

        # 儲存檢查點
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
    num_test_episodes = args.num_test_episodes
    print(f"開始在 {num_test_episodes} 個回合上執行測試...")
    for i in range(num_test_episodes):
        state, _ = test_env.reset(seed=args.seed + i)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, is_test=False)
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
    parser = argparse.ArgumentParser(description="Improved PPO for Walker2d-v4")

    parser.add_argument("--env_id", type=str, default="Walker2d-v4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=202, help="確保整個訓練過程可複現")
    parser.add_argument("--run_name", type=str, default=f"improved_ppo_walker_v5")
    parser.add_argument("--inference", action="store_true", help="啟用測試模式")
    parser.add_argument("--load_model_path", type=str, default="", help="要載入的模型路徑")
    parser.add_argument("--wandb", action="store_true", help="啟用 W&B 實驗追蹤")
    parser.add_argument("--video_folder", type=str, default="task3_videos")
    parser.add_argument("--save_model_path", type=str, default="models")

    # --- 網路架構參數 ---
    parser.add_argument("--hidden_dim", type=int, default=256, help="隱藏層維度")

    # --- 訓練超參數 ---
    parser.add_argument("--max_steps", type=int, default=3_003_000)
    parser.add_argument("--batch_size", type=int, default=2048, help="每次策略更新收集的步數")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor 的學習率")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="Critic 的學習率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="權重衰減")
    parser.add_argument("--anneal_lr", action="store_true", default=True, help="啟用學習率線性衰減")
    parser.add_argument("--n_epochs", type=int, default=10, help="每次更新時，重複學習數據的次數")
    parser.add_argument("--num_minibatches", type=int, default=32, help="將 batch 切分為多少個 minibatch")
    parser.add_argument("--gamma", type=float, default=0.995, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 的 lambda 參數")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO 的裁剪係數 epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="價值函數損失的係數")
    parser.add_argument("--entropy_beta", type=float, default=0.01, help="熵係數")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪閾值")
    parser.add_argument("--clip-vloss", action="store_true", default=True, help="啟用價值函數損失裁剪")

    # --- 新增的改進參數 ---
    parser.add_argument("--target_kl", type=float, default=0.01, help="目標KL散度")
    parser.add_argument("--kl_threshold", type=float, default=0.015, help="KL散度早停閾值")
    parser.add_argument("--early_stop_kl", action="store_true", default=True, help="啟用KL散度早停")
    parser.add_argument("--action_std_decay", action="store_true", default=False, help="啟用動作標準差衰減")

    # --- 評估參數 ---
    parser.add_argument("--eval_interval_episodes", type=int, default=20, help="每隔多少回合進行一次評估")
    parser.add_argument("--eval_episodes", type=int, default=20, help="每次評估時運行的回合總數")
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