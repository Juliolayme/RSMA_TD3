"""
plot_results.py - Ve do thi ket qua huan luyen RSMA DRL

Cach chay:
  python plot_results.py
  python plot_results.py --result-dir results/RSMA_M4_K2_rayleigh
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_moving_average

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, default=None,
                    help='Thu muc chua ket qua. Neu None, tim thu muc moi nhat trong results/')
args = parser.parse_args()


def find_latest_result_dir():
    """Tim thu muc ket qua moi nhat"""
    result_root = 'results'
    if not os.path.exists(result_root):
        raise FileNotFoundError("Khong tim thay thu muc results/")
    dirs = [d for d in os.listdir(result_root)
            if os.path.isdir(os.path.join(result_root, d))]
    if not dirs:
        raise FileNotFoundError("Khong co ket qua nao trong results/")
    # Sap xep theo thoi gian tao
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(result_root, d)), reverse=True)
    return os.path.join(result_root, dirs[0])


def plot_training_results(result_dir):
    """Ve 4 do thi chinh"""
    print(f"Loading results from: {result_dir}")

    # Load data
    rewards = np.load(os.path.join(result_dir, 'episode_rewards.npy'))
    sum_rates = np.load(os.path.join(result_dir, 'episode_sum_rates.npy'))
    common_rates = np.load(os.path.join(result_dir, 'episode_common_rates.npy'))

    episodes = np.arange(1, len(rewards) + 1)
    window = min(20, len(rewards) // 5) if len(rewards) > 5 else 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RSMA DRL Training Results', fontsize=16, fontweight='bold')

    # =========================================
    # Plot 1: Episode Reward
    # =========================================
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > window:
        smooth_rewards = compute_moving_average(rewards, window)
        ax1.plot(np.arange(window, len(rewards) + 1), smooth_rewards,
                 color='blue', linewidth=2, label=f'MA({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('(a) Episode Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # =========================================
    # Plot 2: Sum Rate (bps/Hz)
    # =========================================
    ax2 = axes[0, 1]
    ax2.plot(episodes, sum_rates, alpha=0.3, color='red', label='Raw')
    if len(sum_rates) > window:
        smooth_rates = compute_moving_average(sum_rates, window)
        ax2.plot(np.arange(window, len(sum_rates) + 1), smooth_rates,
                 color='red', linewidth=2, label=f'MA({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Sum Rate (bps/Hz)')
    ax2.set_title('(b) Average Sum Rate per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # =========================================
    # Plot 3: Common Rate vs Private Rate
    # =========================================
    ax3 = axes[1, 0]
    private_rates = sum_rates - common_rates  # Xap xi
    ax3.plot(episodes, common_rates, color='green', alpha=0.5, label='Common Rate')
    ax3.plot(episodes, private_rates, color='orange', alpha=0.5, label='Private Rate (approx)')
    if len(common_rates) > window:
        ax3.plot(np.arange(window, len(common_rates) + 1),
                 compute_moving_average(common_rates, window),
                 color='green', linewidth=2)
        ax3.plot(np.arange(window, len(private_rates) + 1),
                 compute_moving_average(private_rates, window),
                 color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Rate (bps/Hz)')
    ax3.set_title('(c) Common Rate vs Private Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # =========================================
    # Plot 4: Convergence (cumulative max)
    # =========================================
    ax4 = axes[1, 1]
    cummax = np.maximum.accumulate(sum_rates)
    ax4.plot(episodes, cummax, color='purple', linewidth=2, label='Best Sum Rate')
    ax4.fill_between(episodes, 0, cummax, alpha=0.1, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Best Sum Rate (bps/Hz)')
    ax4.set_title('(d) Convergence (Cumulative Best)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Luu figure
    save_path = os.path.join(result_dir, 'training_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_comparison_bar(rsma_rate, noma_rate, sdma_rate, save_dir=None):
    """
    Ve bar chart so sanh RSMA vs NOMA vs SDMA
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['RSMA\n(DRL-TD3)', 'NOMA', 'SDMA\n(ZF)']
    rates = [rsma_rate, noma_rate, sdma_rate]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars = ax.bar(methods, rates, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    # Them gia tri tren moi cot
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Sum Rate Comparison: RSMA vs NOMA vs SDMA', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    # Tim thu muc ket qua
    if args.result_dir:
        result_dir = args.result_dir
    else:
        result_dir = find_latest_result_dir()

    # Ve do thi training
    plot_training_results(result_dir)
