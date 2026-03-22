"""
utils.py - Cac ham tien ich cho RSMA DRL

Bao gom:
- Logging va luu ket qua
- Tinh toan thong ke
- So sanh baseline (NOMA, OMA/SDMA)
"""

import numpy as np
import os
import json
from datetime import datetime


class DataLogger:
    """
    Luu va quan ly du lieu huan luyen
    Tham khao: DataManager trong folder 14
    """

    def __init__(self, save_dir='results', project_name=None):
        if project_name is None:
            project_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.save_dir = os.path.join(save_dir, project_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Du lieu theo episode
        self.episode_rewards = []
        self.episode_sum_rates = []
        self.episode_common_rates = []
        self.episode_avg_splitting = []
        self.episode_power_common_ratio = []

    def log_episode(self, episode, env_history, score):
        """
        Luu thong tin cua 1 episode

        Args:
            episode: so thu tu episode
            env_history: dict history tu environment
            score: tong reward cua episode
        """
        self.episode_rewards.append(score)

        if len(env_history['sum_rate']) > 0:
            self.episode_sum_rates.append(np.mean(env_history['sum_rate']))
            self.episode_common_rates.append(np.mean(env_history['common_rate']))

            # Trung binh splitting ratios trong episode
            all_splits = np.array(env_history['splitting_ratios'])
            self.episode_avg_splitting.append(np.mean(all_splits, axis=0).tolist())

            # Ty le cong suat common / tong
            all_p_c = np.array(env_history['power_common'])
            all_p_priv = np.array(env_history['power_private'])
            total_power = all_p_c + np.sum(all_p_priv, axis=1)
            ratio = np.mean(all_p_c / (total_power + 1e-10))
            self.episode_power_common_ratio.append(ratio)

    def save_results(self):
        """Luu ket qua ra file numpy va JSON"""
        np.save(os.path.join(self.save_dir, 'episode_rewards.npy'),
                np.array(self.episode_rewards))
        np.save(os.path.join(self.save_dir, 'episode_sum_rates.npy'),
                np.array(self.episode_sum_rates))
        np.save(os.path.join(self.save_dir, 'episode_common_rates.npy'),
                np.array(self.episode_common_rates))

        # Luu summary dang JSON
        summary = {
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': float(np.mean(self.episode_rewards[-20:])),
            'final_avg_sum_rate': float(np.mean(self.episode_sum_rates[-20:])),
            'best_sum_rate': float(np.max(self.episode_sum_rates)) if self.episode_sum_rates else 0,
        }
        with open(os.path.join(self.save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {self.save_dir}")

    def save_meta(self, meta_dict):
        """Luu metadata (tham so he thong, hyperparameters)"""
        # Chuyen numpy types sang Python types de JSON serialize duoc
        clean_dict = {}
        for k, v in meta_dict.items():
            if isinstance(v, (np.integer,)):
                clean_dict[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_dict[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_dict[k] = v.tolist()
            else:
                clean_dict[k] = v

        with open(os.path.join(self.save_dir, 'meta.json'), 'w') as f:
            json.dump(clean_dict, f, indent=2)


def compute_moving_average(data, window=20):
    """Tinh moving average de smooth do thi"""
    if len(data) < window:
        window = max(1, len(data))
    return np.convolve(data, np.ones(window) / window, mode='valid')


# =============================================================================
# Baseline: NOMA (Non-Orthogonal Multiple Access)
# =============================================================================

def compute_noma_sum_rate(H, P_max, noise_power, K):
    """
    Tinh sum rate cua NOMA (baseline so sanh)

    NOMA: Superposition coding + SIC
    User duoc sap xep theo ||h_k||^2 tang dan
    User yeu nhat giai ma truoc (khong SIC)
    User manh nhat giai ma sau cung (SIC tat ca user yeu hon)

    Args:
        H: ma tran kenh (M, K)
        P_max: tong cong suat
        noise_power: cong suat nhieu
        K: so users

    Returns:
        sum_rate: tong rate NOMA
    """
    M = H.shape[0]

    # Tinh channel gain cua moi user
    channel_gains = np.array([np.linalg.norm(H[:, k])**2 for k in range(K)])

    # Sap xep user theo channel gain (yeu -> manh)
    sorted_idx = np.argsort(channel_gains)

    # Phan bo cong suat: user yeu duoc nhieu cong suat hon (NOMA principle)
    # Dung phan bo theo ty le nghich: p_k proportional to 1/gain_k
    inv_gains = 1.0 / (channel_gains[sorted_idx] + 1e-10)
    power_ratios = inv_gains / np.sum(inv_gains)
    powers = power_ratios * P_max

    # MRT beamforming cho moi user
    W = np.zeros((M, K), dtype=complex)
    for k in range(K):
        h_k = H[:, k]
        W[:, k] = h_k / (np.linalg.norm(h_k) + 1e-10)

    # Tinh rate voi SIC
    sum_rate = 0.0
    for i, k in enumerate(sorted_idx):
        h_k = H[:, k:k+1]
        w_k = W[:, k:k+1]

        # Signal
        sig = powers[i] * np.abs(h_k.conj().T @ w_k).item()**2

        # Interference tu cac user manh hon (chua SIC duoc)
        interf = 0.0
        for j in range(i + 1, K):
            kk = sorted_idx[j]
            w_kk = W[:, kk:kk+1]
            interf += powers[j] * np.abs(h_k.conj().T @ w_kk).item()**2

        sinr = sig / (interf + noise_power)
        sum_rate += np.log2(1 + sinr)

    return sum_rate


# =============================================================================
# Baseline: SDMA (Space Division Multiple Access) / OMA
# =============================================================================

def compute_sdma_sum_rate(H, P_max, noise_power, K):
    """
    Tinh sum rate cua SDMA voi ZF beamforming (baseline)

    SDMA/OMA: Moi user duoc phuc vu tren khong gian rieng (ZF triet nhieu)

    Args:
        H: ma tran kenh (M, K)
        P_max: tong cong suat
        noise_power: cong suat nhieu
        K: so users

    Returns:
        sum_rate: tong rate SDMA
    """
    M = H.shape[0]

    if K > M:
        # ZF khong kha thi khi K > M, dung MRT
        W = H / np.linalg.norm(H, axis=0, keepdims=True)
    else:
        # ZF precoding
        HH = H.conj().T @ H
        reg = 1e-6 * np.eye(K)
        W = H @ np.linalg.inv(HH + reg)
        # Normalize
        for k in range(K):
            W[:, k] /= (np.linalg.norm(W[:, k]) + 1e-10)

    # Chia deu cong suat
    power_per_user = P_max / K

    sum_rate = 0.0
    for k in range(K):
        h_k = H[:, k:k+1]
        w_k = W[:, k:k+1]

        sig = power_per_user * np.abs(h_k.conj().T @ w_k).item()**2

        interf = 0.0
        for j in range(K):
            if j != k:
                w_j = W[:, j:j+1]
                interf += power_per_user * np.abs(h_k.conj().T @ w_j).item()**2

        sinr = sig / (interf + noise_power)
        sum_rate += np.log2(1 + sinr)

    return sum_rate
