"""
environment.py - Moi truong RSMA (Rate Splitting Multiple Access) cho DRL

He thong: BS (M anten) phuc vu K users (1 anten moi user)
Ky thuat: RSMA - chia tin nhan thanh common message va private messages

Bien toi uu (DRL output):
  1) He so phan chia ty le (splitting ratios): c_k ∈ [0, 1], sum(c_k) <= 1
  2) He so phan bo cong suat: p_c (common), p_k (private), sum <= P_max

Beamforming: tinh closed-form (MRT cho common, ZF cho private)

Tham khao pattern: folder 11 (RIS_MISO), folder 14 (MiniSystem)
"""

import numpy as np
import math
from channel import ChannelModel, dBm_to_watt, watt_to_dBm


class RSMA_Env:
    """
    Moi truong RSMA cho Deep Reinforcement Learning

    RSMA Protocol:
    1) BS chia message cua user k thanh: common part (W_c) + private part (W_k)
    2) Tat ca common parts duoc gop va encode thanh 1 common stream
    3) Moi user giai ma common stream truoc (SIC), roi giai ma private stream
    4) Rate user k = c_k * R_c + R_{p,k}  (voi sum(c_k) = 1)
       voi R_c = min_k{R_{c,k}} la common rate (bi gioi han boi user yeu nhat)
    5) Sum rate = R_c + sum(R_{p,k})     (khi sum(c_k) = 1)
    """

    def __init__(self, M=4, K=2, P_max_dBm=30, noise_power_dBm=-80,
                 channel_type='rayleigh', rician_factor=10.0,
                 user_distances=None, frequency=2.4e9, path_loss_exp=3.0,
                 time_varying=False, channel_correlation=0.9,
                 step_num=100,
                 # === RSMA reward shaping ===
                 lambda_ratio=2.0,        # He so cho common_ratio = R_c/R_sum
                 lambda_log=0.5,          # He so cho log(1 + 5*R_c)
                 collapse_threshold=0.05, # Nguong R_c de phat collapse
                 collapse_penalty=2.0,    # Gia tri phat khi R_c < threshold
                 # === Power floor ===
                 min_power_common_ratio=0.10,  # Toi thieu 10% P_max cho common
                 ):
        """
        Args:
            M: so anten BS (phat)
            K: so users (moi user 1 anten thu)
            P_max_dBm: cong suat phat toi da (dBm)
            noise_power_dBm: cong suat nhieu AWGN (dBm)
            channel_type: loai kenh ('rayleigh' hoac 'rician')
            rician_factor: he so Rician (chi dung khi rician)
            user_distances: khoang cach BS-user (m), shape (K,)
            frequency: tan so song mang (Hz)
            path_loss_exp: he so suy hao
            time_varying: kenh co thay doi theo thoi gian khong
            channel_correlation: he so tuong quan thoi gian (Jake's model)
            step_num: so buoc moi episode
        """
        # === Tham so he thong ===
        self.M = M
        self.K = K
        self.P_max = dBm_to_watt(P_max_dBm)
        self.P_max_dBm = P_max_dBm
        self.noise_power = dBm_to_watt(noise_power_dBm)
        self.noise_power_dBm = noise_power_dBm
        self.time_varying = time_varying
        self.channel_correlation = channel_correlation
        self.step_num = step_num

        # === RSMA reward shaping ===
        self.lambda_ratio = lambda_ratio
        self.lambda_log = lambda_log
        self.collapse_threshold = collapse_threshold
        self.collapse_penalty = collapse_penalty

        # === Power floor ===
        self.min_power_common_ratio = min_power_common_ratio

        # === Tham so kenh ===
        self.channel_params = {
            'M': M, 'K': K,
            'channel_type': channel_type,
            'rician_factor': rician_factor,
            'frequency': frequency,
            'path_loss_exp': path_loss_exp,
            'user_distances': user_distances
        }

        # === Kich thuoc state va action ===
        # State: kenh (2*M*K) + rate truoc (K) + splitting ratios (K) + power (K+1)
        self.state_dim = 2 * M * K + K + K + (K + 1)
        # Action: splitting ratios (K) + power allocation (K+1: 1 common + K private)
        self.action_dim = K + (K + 1)

        # === Khoi tao ===
        self.channel_model = None
        self.H = None  # Ma tran kenh C^{M x K}

        # Bien toi uu
        self.c = None          # Splitting ratios, shape (K,)
        self.p_c = None        # Common power (Watt)
        self.p_private = None  # Private powers, shape (K,)

        # Beamforming vectors (closed-form)
        self.w_c = None    # Common beamformer, shape (M, 1)
        self.W_p = None    # Private beamformers, shape (M, K)

        # Tracking
        self.prev_rates = None
        self.step_count = 0

        # History logging
        self.history = {
            'sum_rate': [],
            'common_rate': [],
            'private_rates': [],
            'splitting_ratios': [],
            'power_common': [],
            'power_private': [],
        }

        self.reset()

    def reset(self):
        """
        Reset moi truong: tao kenh moi, khoi tao lai bien toi uu
        Returns: state vector (numpy array)
        """
        # 1. Tao kenh moi
        self.channel_model = ChannelModel(**self.channel_params)
        self.H = self.channel_model.get_channel_matrix()

        # 2. Khoi tao splitting ratios va power
        self.c = np.ones(self.K) / self.K           # c_k = 1/K
        self.p_c = self.P_max * 0.3                 # 30% cho common
        self.p_private = np.ones(self.K) * (self.P_max * 0.7 / self.K)  # 70% chia deu

        # 3. Tinh beamforming ban dau
        self._compute_beamforming()

        # 4. Reset tracking
        self.prev_rates = np.zeros(self.K)
        self.step_count = 0

        # 5. Clear history
        for key in self.history:
            self.history[key] = []

        return self._get_state()

    def _get_state(self):
        """
        Xay dung vector state tu cac tham so he thong

        State bao gom:
        - Kenh truyen (2*M*K phan tu: real + imag)
        - Rate truoc do (K phan tu)
        - Splitting ratios hien tai (K phan tu)
        - Power allocation hien tai (K+1 phan tu, normalized)

        Returns: numpy array shape (state_dim,)
        """
        # 1. Kenh: tach real va imag
        channel_state = self.channel_model.get_channel_state_vector()

        # 2. Rate truoc do (normalized)
        rate_state = self.prev_rates / max(np.max(np.abs(self.prev_rates)), 1e-6)

        # 3. Splitting ratios (da trong [0, 1])
        split_state = self.c.copy()

        # 4. Power (normalized boi P_max)
        power_state = np.concatenate([[self.p_c], self.p_private]) / self.P_max

        state = np.concatenate([channel_state, rate_state, split_state, power_state])
        return state.astype(np.float32)

    def _compute_beamforming(self):
        """
        Tinh beamforming vectors (closed-form):
        - Common: MRT (Maximum Ratio Transmission) huong toi tong kenh
        - Private: ZF (Zero-Forcing) de triet nhieu giua cac user

        Tham khao: Paper 1 (MRT cho HD mode)
        """
        H = self.H  # (M, K)
        M, K = self.M, self.K

        # === Common beamformer: Normalized MRT ===
        # w_c = sum(h_k / ||h_k||) / || ... ||
        #
        # Tai sao KHONG dung MRT thong thuong (sum(h_k)):
        #   - User gan BS co ||h_k|| lon → chiem uu the trong w_c
        #   - User xa co ||h_k|| nho → beamformer khong huong den no
        #   - R_c = min_k(R_ck) → user xa bi "keo chim" → R_c ≈ 0
        #
        # Normalized MRT: moi user dong gop HUONG (direction) nhu nhau
        #   → Beamformer can bang giua tat ca users
        #   → min(R_ck) cao hon → R_c > 0
        # ================================================================
        H_norms = np.linalg.norm(H, axis=0, keepdims=True)  # (1, K)
        H_norms = np.maximum(H_norms, 1e-10)
        H_unit = H / H_norms                                # (M, K), moi cot la unit vector
        h_sum = np.sum(H_unit, axis=1, keepdims=True)        # (M, 1)
        norm_h_sum = np.linalg.norm(h_sum)
        if norm_h_sum > 1e-10:
            self.w_c = h_sum / norm_h_sum
        else:
            self.w_c = np.ones((M, 1), dtype=complex) / np.sqrt(M)

        # === Private beamformers: ZF ===
        if K <= M:
            # ZF: W_p = H (H^H H)^{-1}
            HH_H = H.conj().T @ H  # (K, K)
            try:
                # Regularized inverse de tranh singular
                reg = 1e-6 * np.eye(K)
                W_zf = H @ np.linalg.inv(HH_H + reg)  # (M, K)
                # Normalize moi cot
                for k in range(K):
                    col_norm = np.linalg.norm(W_zf[:, k])
                    if col_norm > 1e-10:
                        W_zf[:, k] /= col_norm
                self.W_p = W_zf
            except np.linalg.LinAlgError:
                # Fallback ve MRT neu ZF that bai
                self.W_p = H / np.linalg.norm(H, axis=0, keepdims=True)
        else:
            # Khi K > M: khong the ZF, dung MRT
            self.W_p = H / np.linalg.norm(H, axis=0, keepdims=True)

    def _compute_sinr_common(self, k):
        """
        Tinh SINR cua common message tai user k

        SINR_{c,k} = p_c * |h_k^H w_c|^2 / (sum_j p_j * |h_k^H w_j|^2 + sigma^2)

        User k giai ma common message truoc, nen interference chi tu private streams
        """
        h_k = self.H[:, k:k+1]  # (M, 1)

        # Signal power: common stream
        sig_common = self.p_c * np.abs(h_k.conj().T @ self.w_c).item() ** 2

        # Interference: tat ca private streams (chua SIC)
        interf = 0.0
        for j in range(self.K):
            w_j = self.W_p[:, j:j+1]
            interf += self.p_private[j] * np.abs(h_k.conj().T @ w_j).item() ** 2

        sinr = sig_common / (interf + self.noise_power)
        return sinr

    def _compute_sinr_private(self, k):
        """
        Tinh SINR cua private message tai user k (sau khi SIC bo common)

        SINR_{p,k} = p_k * |h_k^H w_k|^2 / (sum_{j!=k} p_j * |h_k^H w_j|^2 + sigma^2)

        Sau SIC: common stream da bi loai bo, chi con nhieu tu private streams khac
        """
        h_k = self.H[:, k:k+1]  # (M, 1)

        # Signal power: private stream cua user k
        w_k = self.W_p[:, k:k+1]
        sig_private = self.p_private[k] * np.abs(h_k.conj().T @ w_k).item() ** 2

        # Interference: private streams cua cac user khac
        interf = 0.0
        for j in range(self.K):
            if j != k:
                w_j = self.W_p[:, j:j+1]
                interf += self.p_private[j] * np.abs(h_k.conj().T @ w_j).item() ** 2

        sinr = sig_private / (interf + self.noise_power)
        return sinr

    def _compute_rates(self):
        """
        Tinh rate theo RSMA protocol:

        1) R_{c,k} = log2(1 + SINR_{c,k})    -- rate common tai user k
        2) R_c = min_k{R_{c,k}}               -- common rate (bottleneck)
        3) R_{p,k} = log2(1 + SINR_{p,k})     -- private rate user k
        4) R_k = c_k * R_c + R_{p,k}          -- tong rate user k

        Rang buoc: sum(c_k) <= 1  (tong ty le chia common rate)

        Returns:
            R_total: shape (K,) - tong rate moi user
            R_c: scalar - common rate
            R_private: shape (K,) - private rate moi user
        """
        R_c_per_user = np.zeros(self.K)
        R_private = np.zeros(self.K)

        for k in range(self.K):
            # Common rate tai user k
            sinr_c = self._compute_sinr_common(k)
            R_c_per_user[k] = np.log2(1 + sinr_c)

            # Private rate tai user k (sau SIC)
            sinr_p = self._compute_sinr_private(k)
            R_private[k] = np.log2(1 + sinr_p)

        # Common rate bi gioi han boi user yeu nhat (bottleneck)
        R_c = np.min(R_c_per_user)

        # Tong rate moi user
        R_total = np.zeros(self.K)
        for k in range(self.K):
            R_total[k] = self.c[k] * R_c + R_private[k]

        return R_total, R_c, R_private

    def step(self, action):
        """
        Thuc hien 1 buoc trong moi truong

        Args:
            action: numpy array shape (action_dim,), output cua Actor (∈ [-1, 1] do tanh)
                - action[:K]   -> raw splitting ratios
                - action[K:]   -> raw power allocation (K+1 gia tri)

        Returns:
            next_state: numpy array shape (state_dim,)
            reward: scalar (sum rate)
            done: bool
            info: dict chua thong tin chi tiet
        """
        K = self.K
        self.step_count += 1

        # ============================================
        # 1. Parse action -> bien vat ly
        # ============================================

        # ===========================================================
        # 1a) Splitting ratios: SOFTMAX de dam bao sum(c_k) = 1
        #     (FIX A: cu dung sigmoid + "<=1" -> lang phi common rate)
        #
        #     Theo RSMA paper: sum(c_k) PHAI = 1
        #     De R_sum = R_c + sum(R_p_k) (khong lang phi)
        # ===========================================================
        raw_c = action[:K]
        raw_c_clipped = np.clip(raw_c, -10, 10)
        raw_c_shifted = raw_c_clipped - np.max(raw_c_clipped)
        exp_c = np.exp(raw_c_shifted)
        self.c = exp_c / np.sum(exp_c)  # softmax -> sum = 1.0 chinh xac

        # ===========================================================
        # 1b) Power allocation: map [-1,1] -> [0,1] rồi normalize
        #     (FIX C: bo softmax+bias hacky)
        #
        #     Actor output tanh ∈ [-1, 1]
        #     -> (x+1)/2 ∈ [0, 1] -> normalize * P_max
        # ===========================================================
        raw_p = action[K:]  # K+1 phan tu: [common, private_1, ..., private_K]
        p_positive = (raw_p + 1.0) / 2.0           # map [-1,1] -> [0,1]
        p_positive = np.maximum(p_positive, 1e-6)   # tranh chia 0

        # Normalize de tong = P_max
        p_ratio = p_positive / np.sum(p_positive)
        self.p_c = p_ratio[0] * self.P_max
        self.p_private = p_ratio[1:] * self.P_max

        # Enforce minimum common power SAU normalize
        min_pc = self.min_power_common_ratio * self.P_max
        if self.p_c < min_pc:
            self.p_c = min_pc
            remaining = self.P_max - min_pc
            pp_sum = np.sum(self.p_private)
            if pp_sum > 1e-10:
                self.p_private = self.p_private / pp_sum * remaining
            else:
                self.p_private = np.ones(K) * remaining / K

        # ============================================
        # 2. Cap nhat kenh (neu time-varying)
        # ============================================
        if self.time_varying:
            self.channel_model.update_channel(self.channel_correlation)
            self.H = self.channel_model.get_channel_matrix()

        # ============================================
        # 3. Tinh beamforming (closed-form)
        # ============================================
        self._compute_beamforming()

        # ============================================
        # 4. Tinh rate va reward
        # ============================================
        R_total, R_c, R_private = self._compute_rates()

        # R_sum = R_c + sum(R_p_k) vi sum(c_k)=1
        R_sum_private = np.sum(R_private)
        R_sum = np.sum(R_total)   # = R_c + R_sum_private (chinh xac)

        # ============ REWARD (Multi-Component) ============
        # Cong thuc:
        #   reward = R_sum
        #          + λ_ratio * (R_c / R_sum)     ... khuyến khích tỷ lệ common
        #          + λ_log   * log(1 + 5·R_c)    ... smooth bonus, không nổ
        #          - penalty  (nếu R_c < threshold)
        #
        # Tai sao tot hon (1+λ)*R_c:
        #   - R_c/R_sum ∈ [0,1]: normalized, gradient stable bat ke R_sum lon/nho
        #   - log(1+5*R_c): tang nhanh khi R_c nho (khuyến khích khám phá),
        #     bão hòa khi R_c lớn (không chiếm ưu thế)
        #   - R_sum: giu muc tieu chinh (maximize throughput)
        # ===================================================
        eps = 1e-6
        common_ratio = R_c / (R_sum + eps)                # ∈ [0, 1]
        common_log_bonus = np.log(1.0 + 5.0 * R_c)       # smooth, > 0

        reward = (R_sum
                  + self.lambda_ratio * common_ratio
                  + self.lambda_log * common_log_bonus)

        # Collapse penalty: phat khi common rate qua nho
        collapse_pen = 0.0
        if R_c < self.collapse_threshold:
            collapse_pen = self.collapse_penalty
            reward -= collapse_pen

        # ============================================
        # 5. Luu tracking
        # ============================================
        self.prev_rates = R_total.copy()

        # FIX D: Luu PURE sum rate (khong phai shaped reward) de log/plot dung
        self.history['sum_rate'].append(R_sum)
        self.history['common_rate'].append(R_c)
        self.history['private_rates'].append(R_private.copy())
        self.history['splitting_ratios'].append(self.c.copy())
        self.history['power_common'].append(self.p_c)
        self.history['power_private'].append(self.p_private.copy())

        # ============================================
        # 6. Kiem tra dieu kien ket thuc
        # ============================================
        done = (self.step_count >= self.step_num)

        # ============================================
        # 7. Tra ve
        # ============================================
        next_state = self._get_state()

        info = {
            'sum_rate': R_sum,                       # Pure R_c + sum(R_p_k)
            'shaped_reward': reward,                 # Reward thuc te (co shaping)
            'common_rate': R_c,                      # R_c = min_k(R_{c,k})
            'sum_private_rate': R_sum_private,       # sum(R_p_k)
            'collapse_penalty': collapse_pen,
            'private_rates': R_private.copy(),       # R_p_k per user
            'user_rates': R_total.copy(),            # c_k*R_c + R_p_k per user
            'splitting_ratios': self.c.copy(),       # c_k, sum=1
            'power_common': self.p_c,                # Pc (Watt)
            'power_private': self.p_private.copy(),  # Pp_k (Watt)
            'power_total': self.p_c + np.sum(self.p_private),
            'power_common_ratio': self.p_c / (self.p_c + np.sum(self.p_private) + 1e-10),
            'step': self.step_count,
        }

        return next_state, reward, done, info

    def get_system_state_dim(self):
        """Tra ve kich thuoc state (tuong thich voi pattern folder 14)"""
        return self.state_dim

    def get_system_action_dim(self):
        """Tra ve kich thuoc action (tuong thich voi pattern folder 14)"""
        return self.action_dim

    def get_system_info(self):
        """In thong tin he thong"""
        info = {
            'M (BS antennas)': self.M,
            'K (users)': self.K,
            'P_max (dBm)': self.P_max_dBm,
            'Noise power (dBm)': self.noise_power_dBm,
            'Channel type': self.channel_params['channel_type'],
            'State dim': self.state_dim,
            'Action dim': self.action_dim,
            'Time-varying': self.time_varying,
        }
        return info
