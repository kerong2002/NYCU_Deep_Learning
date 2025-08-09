import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import wandb
import math


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


def orthogonal_init(layer: nn.Linear, gain: float = math.sqrt(2)):
    """對指定的線性層進行正交初始化。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


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
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)

        orthogonal_init(self.network[0])
        orthogonal_init(self.network[2])
        orthogonal_init(self.mean_head, gain=0.01)

        self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state_tensor: torch.Tensor) -> torch.distributions.Normal:
        x = self.network(state_tensor)
        action_means = self.mean_head(x)
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
        orthogonal_init(self.network[0])
        orthogonal_init(self.network[2])
        orthogonal_init(self.network[4], gain=1.0)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(state_tensor)


# =============================================================================
#  3. Runner: 負責與環境互動
# =============================================================================

class Runner:
    """管理代理人與環境之間的互動，負責收集經驗。"""

    def __init__(self, env: gym.Env, actor: Actor, critic: Critic, max_total_steps: int):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.max_total_steps = max_total_steps
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.state, _ = self.env.reset()
        # 移除 self.done 旗標，改為在迴圈內局部判斷

    def run(self, num_steps: int, device: torch.device) -> dict:
        memory = {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': [], 'values': []
        }

        # ### [修正] ###
        # 將原本在迴圈開頭的 if self.done 檢查塊移除，
        # 改為在 env.step 後立刻判斷並處理回合結束的邏輯。
        # 這樣可以確保回合分數被即時記錄。

        for _ in range(num_steps):
            if self.total_steps >= self.max_total_steps: break

            # 採樣動作
            with torch.no_grad():
                state_tensor = to_tensor(self.state, device).unsqueeze(0)
                policy_dist = self.actor(state_tensor)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            clipped_action = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

            # 與環境互動
            next_state, reward, terminated, truncated, _ = self.env.step(clipped_action)
            done = terminated or truncated

            # 儲存經驗
            memory['states'].append(self.state)
            memory['actions'].append(action_np)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            memory['log_probs'].append(log_prob.cpu().numpy())
            memory['values'].append(value.cpu().numpy().flatten())

            # 更新狀態
            self.state = next_state
            self.total_steps += 1
            self.episode_reward += reward

            # **關鍵修正**：在 env.step 後立刻檢查回合是否結束
            if done:
                self.episode_rewards.append(self.episode_reward)
                print(
                    f"訓練回合 {len(self.episode_rewards)} 結束. 分數: {self.episode_reward:.2f}. 總步數: {self.total_steps}")

                # 立刻重置環境和計分器
                self.state, _ = self.env.reset()
                self.episode_reward = 0

        # 計算最後一個狀態的價值，用於 GAE
        with torch.no_grad():
            last_state_tensor = to_tensor(self.state, device).unsqueeze(0)
            last_value = self.critic(last_state_tensor).cpu().numpy().flatten()

        # last_done 應該反映最後一個狀態是否為終止狀態。
        # 在我們的修正邏輯中，如果迴圈結束時 state 剛被 reset，那它不是終止狀態。
        # 如果迴圈正常結束，我們需要從 memory 中獲取最後一個 done 的值。
        memory['last_value'] = last_value
        memory['last_done'] = memory['dones'][-1] if memory['dones'] else False

        return memory


# =============================================================================
#  4. Learner: 負責模型更新 (PPO 核心)
# =============================================================================

class PPOLearner:
    """管理 PPO 學習過程 (更新 Actor 和 Critic 網路)。"""

    def __init__(self, actor: Actor, critic: Critic, config: argparse.Namespace):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr, eps=1e-5)

    def learn(self, memory: dict, current_step: int, device: torch.device):
        if not memory['states']: return

        b_states = to_tensor(np.array(memory['states']), device)
        b_actions = to_tensor(np.array(memory['actions']), device)
        b_log_probs_old = to_tensor(np.array(memory['log_probs']), device).flatten()
        b_advantages = to_tensor(np.array(memory['advantages']), device).flatten()
        b_returns = to_tensor(np.array(memory['returns']), device).flatten()

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_states.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches
        indices = np.arange(batch_size)

        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_states = b_states[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs_old = b_log_probs_old[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]

                # --- Critic 更新 ---
                new_values = self.critic(mb_states).view(-1)
                v_loss = F.smooth_l1_loss(new_values, mb_returns)

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                # --- Actor 更新 ---
                policy_dist = self.actor(mb_states)
                new_log_probs = policy_dist.log_prob(mb_actions).sum(dim=-1)
                entropy = policy_dist.entropy().mean()

                log_ratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(log_ratio)

                surr1 = mb_advantages * ratio
                surr2 = mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = -torch.min(surr1, surr2).mean()

                actor_loss = pg_loss - self.config.entropy_beta * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

        if wandb.run:
            wandb.log({
                "Train/actor_loss": pg_loss.item(),
                "Train/critic_loss": v_loss.item(),
                "Train/entropy": entropy.item(),
                "Global Step": current_step
            })


# =============================================================================
#  5. GAE 計算與評估函式
# =============================================================================

def compute_gae_and_returns(memory: dict, config: argparse.Namespace):
    """計算廣義優勢估計 (GAE) 和回報 (Returns)。"""
    rewards = np.array(memory['rewards'])
    values = np.array(memory['values']).flatten()
    dones = np.array(memory['dones'])
    last_value = memory['last_value'][0]
    last_done = memory['last_done']

    num_steps = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        delta = rewards[t] + config.gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae_lam

    memory['advantages'] = advantages
    memory['returns'] = advantages + values


def evaluate_during_training(training_env: gym.Env, learner: PPOLearner, config: argparse.Namespace, total_steps: int,
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

### [整合] 以下是從 a2c_pendulum.py 整合而來的完整測試函式 ###
def test_agent(config: argparse.Namespace):
    """
    載入已訓練的模型，並在固定種子上進行測試，同時錄製影片。
    """
    print("\n--- 啟動測試模式 (Inference Mode) ---")
    if not config.load_model_path: raise ValueError("測試模式需要模型路徑，請使用 --load_model_path 指定。")

    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {config.device}")
    print(f"載入模型: {config.load_model_path}")

    # 載入模型檔案，weights_only=False 是為了能載入 obs_rms 這個 numpy 物件
    model_data = torch.load(config.load_model_path, map_location=config.device, weights_only=False)

    # 從儲存的檔案中動態獲取 state 和 action 維度
    state_dim = model_data['obs_rms'].mean.shape[0]
    action_dim = model_data['actor_state_dict']['log_stds'].shape[0]

    # 建立 Actor 模型並載入權重
    actor = Actor(state_dim, action_dim).to(config.device)
    actor.load_state_dict(model_data['actor_state_dict'])
    actor.eval()  # 切換到評估模式

    # 建立測試環境
    base_env = gym.make(config.env_id, render_mode="rgb_array")
    # **關鍵**：使用與訓練時相同的狀態正規化層
    norm_env = gym.wrappers.NormalizeObservation(base_env)
    print("正在還原儲存的觀測狀態正規化 (obs_rms) 物件...")
    # **關鍵**：將訓練時儲存的 running mean/std 還原到正規化層中
    norm_env.obs_rms = model_data['obs_rms']
    norm_env.training = False  # 確保在測試時 running mean/std 不再更新
    print("還原成功！")

    # 設定影片錄製路徑並包裹環境
    video_path = os.path.join(config.video_folder, "inference_run_fixed_seeds")
    test_env = gym.wrappers.RecordVideo(norm_env, video_folder=video_path, episode_trigger=lambda x: True)

    total_reward = 0
    num_test_episodes = 20
    print(f"開始在固定的 {num_test_episodes} 個種子 (707 to {707 + num_test_episodes - 1}) 上執行測試...")

    for i in range(707, 707 + num_test_episodes):
        state, _ = test_env.reset(seed=i)
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = to_tensor(state, config.device).unsqueeze(0)
                # 在測試時，我們通常取機率分佈的平均值作為確定性動作
                action = actor(state_t).mean.squeeze(0).cpu().numpy()

            state, reward, terminated, truncated, _ = test_env.step(
                np.clip(action, test_env.action_space.low, test_env.action_space.high))
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward
        print(f"測試回合 {i - 707 + 1}/{num_test_episodes} (seed={i}), 分數: {episode_reward:.2f}")

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
    parser = argparse.ArgumentParser(description="PPO for Pendulum-v1 (由 A2C 框架修改)")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=707, help="確保整個訓練過程可複現")
    parser.add_argument("--run_name", type=str, default="ppo_from_a2c_framework_v3_fixed")
    parser.add_argument("--inference", action="store_true", help="啟用測試模式")
    parser.add_argument("--load_model_path", type=str, default="", help="要載入的模型路徑")
    parser.add_argument("--wandb", action="store_true", help="啟用 W&B 實驗追蹤")
    parser.add_argument("--video_folder", type=str, default="task2_videos")
    parser.add_argument("--save_model_path", type=str, default="task2_models")

    # --- 訓練參數 (依照您的要求，保持不變) ---
    parser.add_argument("--max_steps", type=int, default=200200)
    parser.add_argument("--batch_size", type=int, default=2000, help="每次策略更新收集的步數 (Rollout Buffer Size)")
    parser.add_argument("--actor_lr", type=float, default=5e-4, help="Actor 的學習率")
    parser.add_argument("--critic_lr", type=float, default=3e-3, help="Critic 的學習率")
    parser.add_argument("--n_epochs", type=int, default=20, help="每次更新時，重複學習數據的次數")
    parser.add_argument("--num_minibatches", type=int, default=32, help="將 batch 切分為多少個 minibatch")
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 的 lambda 參數")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO 的裁剪係數 epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="價值函數損失的係數")
    parser.add_argument("--entropy_beta", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # --- 評估參數 ---
    parser.add_argument("--eval_interval_episodes", type=int, default=20, help="每隔多少回合進行一次評估")
    parser.add_argument("--eval_episodes", type=int, default=20, help="每次評估時運行的回合總數")

    return parser.parse_args()


def main(config: argparse.Namespace):
    ### [整合] 根據 --inference 參數決定執行訓練或測試 ###
    if config.inference:
        test_agent(config)
    else:
        train_agent(config)


def train_agent(config: argparse.Namespace):
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"--- 啟動 PPO 訓練模式 ---")
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
    learner = PPOLearner(actor, critic, config)
    runner = Runner(env, actor, critic, config.max_steps)

    if config.wandb:
        try:
            wandb.init(project=f"PPO-{config.env_id}", name=config.run_name, config=config)
            wandb.define_metric("Global Step")
            wandb.define_metric("Train/*", step_metric="Global Step")
            wandb.define_metric("Eval/*", step_metric="Global Step")
        except Exception as e:
            print(f"Wandb 初始化失敗: {e}，將在無 wandb 模式下繼續。")

    best_eval_score = -np.inf
    last_eval_episode = 0

    print("\n--- 開始 PPO 訓練迴圈 (使用固定種子) ---")
    while runner.total_steps < config.max_steps:
        # 實現學習率線性衰減
        frac = 1.0 - (runner.total_steps - 1.0) / config.max_steps
        lr_now = frac * config.actor_lr
        learner.actor_optimizer.param_groups[0]["lr"] = lr_now
        lr_now = frac * config.critic_lr
        learner.critic_optimizer.param_groups[0]["lr"] = lr_now

        memory = runner.run(config.batch_size, config.device)
        if not memory['states']: break

        compute_gae_and_returns(memory, config)
        learner.learn(memory, runner.total_steps, config.device)

        current_episode_count = len(runner.episode_rewards)
        if current_episode_count >= last_eval_episode + config.eval_interval_episodes:
            avg_reward, best_eval_score = evaluate_during_training(env, learner, config, runner.total_steps,
                                                                   best_eval_score)
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
    main(args)

'''
!/usr/bin/env python3
-*- coding: utf-8 -*-
Pendulum-v1 的 PPO 演算法實現 (由 A2C 框架修改而來)

此版本基於用戶提供的 A2C 框架，將其升級為 PPO，旨在提升訓練穩定性和收斂速度。

核心升級:
1. Runner/Learner 結構保留，職責依然清晰。
2. **關鍵升級**: 引入 GAE (Generalized Advantage Estimation) 來更準確地估計優勢。
3. **關鍵升級**: Learner 實現 PPO 的核心機制：多次更新 (Epochs)、小批量 (Minibatches) 和裁剪目標函數。
4. **關鍵升級**: Runner 現在會收集 log_probs 和 values，以支持 PPO 的計算。
5. 保留了原始程式碼的所有優點，如狀態正規化、固定種子、W&B 整合等。

v2.0 修復:
- 修正了 GAE 計算中的維度問題，解決了 DeprecationWarning 並提升了訓練穩定性。
- 將評估觸發機制從「按步數」改為「按回合數」，以符合用戶需求。

v3.0 (本次修改):
- **[修正]** 為 Actor 和 Critic 分離優化器，以實現更穩定的訓練。
- **[增強]** 引入學習率線性衰減 (LR Annealing)，幫助模型在訓練後期更好地收斂。
- **[增強]** 為神經網路加入正交初始化 (Orthogonal Initialization)，此為 PPO 的常用技巧。
- **[優化]** 將 Critic 損失函數改為 Smooth L1 Loss (Huber Loss)，以增強對離群值的穩健性。
- **[優化]** 將優勢正規化 (Advantage Normalization) 的位置調整到更標準的位置。
- **[優化]** 將 Actor 損失函數的寫法調整為更主流的形式，提高可讀性。

v4.0 (本次修改):
- **[整合]** 根據您的要求，從 a2c_pendulum.py 腳本中學習並整合了完整的獨立測試 (Inference) 功能。
'''