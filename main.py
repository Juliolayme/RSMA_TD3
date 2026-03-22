"""
main.py - Script huan luyen TD3 cho toi uu RSMA

Toi uu dong thoi:
  1) He so phan chia ty le RSMA (splitting ratios c_k)
  2) He so phan bo cong suat (p_c, p_1, ..., p_K)

Cau truc code theo pattern folder 14 (main_train.py)

Cach chay:
  python main.py
  python main.py --M 4 --K 3 --episodes 500
  python main.py --channel rician --time-varying
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import math
import time
import torch

from environment import RSMA_Env
from td3 import Agent
from utils import DataLogger, compute_noma_sum_rate, compute_sdma_sum_rate

# =============================================================================
# 1. Parse arguments
# =============================================================================
parser = argparse.ArgumentParser(description='RSMA DRL Optimization with TD3')
parser.add_argument('--M', type=int, default=4, help='So anten BS (default: 4)')
parser.add_argument('--K', type=int, default=2, help='So users (default: 2)')
parser.add_argument('--P-max', type=float, default=30, help='Cong suat toi da dBm (default: 30)')
parser.add_argument('--noise', type=float, default=-80, help='Cong suat nhieu dBm (default: -80)')
parser.add_argument('--channel', type=str, default='rayleigh', choices=['rayleigh', 'rician'],
                    help='Loai kenh (default: rayleigh)')
parser.add_argument('--time-varying', action='store_true', help='Kenh thay doi theo thoi gian')
parser.add_argument('--episodes', type=int, default=300, help='So episodes (default: 300)')
parser.add_argument('--steps', type=int, default=100, help='So steps moi episode (default: 100)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--project', type=str, default=None, help='Ten project de luu ket qua')
args = parser.parse_args()

# =============================================================================
# 2. Set random seed
# =============================================================================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# =============================================================================
# 3. Khoi tao moi truong RSMA
# =============================================================================
env = RSMA_Env(
    M=args.M,
    K=args.K,
    P_max_dBm=args.P_max,
    noise_power_dBm=args.noise,
    channel_type=args.channel,
    time_varying=args.time_varying,
    step_num=args.steps
)

# =============================================================================
# 4. Khoi tao TD3 Agent (theo pattern folder 14)
# =============================================================================
episode_num = args.episodes
step_num = args.steps

# Hyperparameters (tuong tu folder 14)
agent_param = {
    'alpha': 0.0001,          # Actor learning rate
    'beta': 0.001,            # Critic learning rate
    'input_dims': env.get_system_state_dim(),
    'tau': 0.001,             # Soft update factor
    'batch_size': 64,
    'n_actions': env.get_system_action_dim(),
    'action_noise_factor': 0.3,  # Noise cho exploration
    'memory_max_size': int(episode_num * step_num),
    'agent_name': 'RSMA',
    # Layer sizes (nho hon folder 14 vi bai toan don gian hon)
    'layer1_size': 400,
    'layer2_size': 300,
    'layer3_size': 256,
    'layer4_size': 128,
}

agent = Agent(
    alpha=agent_param['alpha'],
    beta=agent_param['beta'],
    input_dims=[agent_param['input_dims']],
    tau=agent_param['tau'],
    env=env,
    batch_size=agent_param['batch_size'],
    layer1_size=agent_param['layer1_size'],
    layer2_size=agent_param['layer2_size'],
    layer3_size=agent_param['layer3_size'],
    layer4_size=agent_param['layer4_size'],
    n_actions=agent_param['n_actions'],
    max_size=agent_param['memory_max_size'],
    agent_name=agent_param['agent_name']
)

# =============================================================================
# 5. Khoi tao Logger
# =============================================================================
project_name = args.project or f"RSMA_M{args.M}_K{args.K}_{args.channel}"
logger = DataLogger(save_dir='results', project_name=project_name)

# Luu metadata
meta = {
    'system': env.get_system_info(),
    'agent': {k: v for k, v in agent_param.items() if k != 'agent_name'},
    'episodes': episode_num,
    'steps': step_num,
    'seed': args.seed,
}
logger.save_meta(meta)

# =============================================================================
# 6. In thong tin he thong
# =============================================================================
print("=" * 60)
print("       RSMA DRL Optimization with TD3")
print("=" * 60)
print(f"  BS antennas (M):      {args.M}")
print(f"  Users (K):            {args.K}")
print(f"  P_max:                {args.P_max} dBm")
print(f"  Noise power:          {args.noise} dBm")
print(f"  Channel:              {args.channel}")
print(f"  Time-varying:         {args.time_varying}")
print(f"  Episodes:             {episode_num}")
print(f"  Steps/episode:        {step_num}")
print(f"  State dim:            {env.state_dim}")
print(f"  Action dim:           {env.action_dim}")
print(f"    - Splitting ratios: {args.K} dims")
print(f"    - Power allocation: {args.K + 1} dims (1 common + {args.K} private)")
print(f"  Device:               {agent.actor.device}")
print("=" * 60)

# =============================================================================
# 7. Training Loop (theo pattern folder 14)
# =============================================================================
best_score = -np.inf
start_time = time.time()

for episode_cnt in range(episode_num):
    # 1. Reset moi truong
    observation = env.reset()
    step_cnt = 0
    score_per_ep = 0

    while step_cnt < step_num:
        step_cnt += 1

        # 2. Chon action voi noise giam dan (greedy decay)
        greedy = agent_param['action_noise_factor'] * \
                 math.pow((1 - episode_cnt / episode_num), 2)
        action = agent.choose_action(observation, greedy=greedy)

        # 3. Thuc hien action trong moi truong
        new_state, reward, done, info = env.step(action)

        score_per_ep += reward

        # 4. Luu transition vao replay buffer
        agent.remember(observation, action, reward, new_state, int(done))

        # 5. Hoc tu replay buffer
        agent.learn()

        observation = new_state

        if done:
            break

    # 6. Log ket qua episode
    logger.log_episode(episode_cnt, env.history, score_per_ep)

    # 7. In ket qua moi 10 episodes
    if (episode_cnt + 1) % 10 == 0:
        avg_rate = np.mean(env.history['sum_rate']) if env.history['sum_rate'] else 0
        last_split = info['splitting_ratios']
        last_pc_ratio = info['power_common'] / (info['power_total'] + 1e-10)

        elapsed = time.time() - start_time
        print(f"Ep {episode_cnt + 1:>4d}/{episode_num} | "
              f"Score: {score_per_ep:>8.2f} | "
              f"Avg Rate: {avg_rate:>6.3f} bps/Hz | "
              f"Split: [{', '.join(f'{s:.2f}' for s in last_split)}] | "
              f"P_c ratio: {last_pc_ratio:.2f} | "
              f"Noise: {greedy:.3f} | "
              f"Time: {elapsed:.0f}s")

    # 8. Luu model moi 50 episodes
    if (episode_cnt + 1) % 50 == 0:
        agent.save_models()

    # 9. Track best score
    if score_per_ep > best_score:
        best_score = score_per_ep

# =============================================================================
# 8. Luu ket qua cuoi cung
# =============================================================================
agent.save_models()
logger.save_results()

total_time = time.time() - start_time
print("\n" + "=" * 60)
print("  Training Complete!")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Best score: {best_score:.4f}")
print(f"  Final avg sum rate: {np.mean(logger.episode_sum_rates[-20:]):.4f} bps/Hz")
print("=" * 60)

# =============================================================================
# 9. So sanh Baseline (NOMA va SDMA)
# =============================================================================
print("\n--- Baseline Comparison (last channel realization) ---")
H = env.H
P_max = env.P_max
noise = env.noise_power
K = env.K

rsma_rate = np.mean(logger.episode_sum_rates[-20:])
noma_rate = compute_noma_sum_rate(H, P_max, noise, K)
sdma_rate = compute_sdma_sum_rate(H, P_max, noise, K)

print(f"  RSMA (DRL):   {rsma_rate:.4f} bps/Hz")
print(f"  NOMA:         {noma_rate:.4f} bps/Hz")
print(f"  SDMA (ZF):    {sdma_rate:.4f} bps/Hz")
print(f"  RSMA gain vs NOMA: {(rsma_rate/noma_rate - 1)*100:.1f}%" if noma_rate > 0 else "")
print(f"  RSMA gain vs SDMA: {(rsma_rate/sdma_rate - 1)*100:.1f}%" if sdma_rate > 0 else "")
