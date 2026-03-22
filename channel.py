"""
channel.py - Mo hinh kenh truyen cho he thong RSMA
Tham khao: folder 14 (channel.py) va folder 11 (environment.py)

Ho tro:
- Rayleigh fading (NLOS)
- Rician fading (LOS + NLOS)
- Path loss model
- Time-varying channel (tuy chon)
"""

import numpy as np
import math


def dB_to_linear(dB_value):
    """Chuyen tu dB sang gia tri tuyen tinh"""
    return 10 ** (dB_value / 10.0)


def dBm_to_watt(dBm_value):
    """Chuyen tu dBm sang Watt"""
    return 10 ** ((dBm_value - 30) / 10.0)


def watt_to_dBm(watt_value):
    """Chuyen tu Watt sang dBm"""
    if watt_value <= 0:
        return -np.inf
    return 10 * np.log10(watt_value) + 30


def compute_path_loss(distance, frequency=2.4e9, path_loss_exp=3.0, ref_distance=1.0):
    """
    Tinh path loss theo mo hinh log-distance

    Args:
        distance: khoang cach (m)
        frequency: tan so song mang (Hz)
        path_loss_exp: he so suy hao (alpha)
        ref_distance: khoang cach tham chieu (m)

    Returns:
        path_loss: gia tri path loss (tuyen tinh, khong phai dB)
    """
    c = 3e8  # toc do anh sang
    wavelength = c / frequency
    # Free-space path loss tai khoang cach tham chieu
    PL_0 = (4 * math.pi * ref_distance / wavelength) ** 2
    # Log-distance path loss
    if distance < ref_distance:
        distance = ref_distance
    PL = PL_0 * (distance / ref_distance) ** path_loss_exp
    return PL


class ChannelModel:
    """
    Mo hinh kenh truyen MISO (Multiple Input Single Output)

    Tao kenh tu BS (M anten) den moi user (1 anten)
    Ho tro: Rayleigh, Rician
    """

    def __init__(self, M, K, channel_type='rayleigh', rician_factor=10.0,
                 frequency=2.4e9, path_loss_exp=3.0, user_distances=None):
        """
        Args:
            M: so anten BS
            K: so users
            channel_type: 'rayleigh' hoac 'rician'
            rician_factor: he so Rician K_r (chi dung khi channel_type='rician')
            frequency: tan so song mang (Hz)
            path_loss_exp: he so suy hao duong truyen
            user_distances: khoang cach tu BS den moi user (m), shape (K,)
        """
        self.M = M
        self.K = K
        self.channel_type = channel_type
        self.rician_factor = rician_factor
        self.frequency = frequency
        self.path_loss_exp = path_loss_exp

        # Khoang cach mac dinh neu khong truyen vao
        if user_distances is None:
            self.user_distances = np.random.uniform(50, 200, size=K)
        else:
            self.user_distances = np.array(user_distances)

        # Tinh path loss cho moi user
        self.path_loss = np.array([
            compute_path_loss(d, frequency, path_loss_exp)
            for d in self.user_distances
        ])

        # Khoi tao kenh
        self.H = None  # Ma tran kenh H: shape (M, K), cot k la h_k
        self.generate_channel()

    def generate_channel(self):
        """
        Tao ma tran kenh H ∈ C^{M x K}
        Cot k cua H la vector kenh h_k ∈ C^{M x 1} tu BS den user k
        """
        M, K = self.M, self.K

        if self.channel_type == 'rayleigh':
            # Rayleigh fading: h_k ~ CN(0, I/PL_k)
            H_iid = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)
            # Ap dung path loss
            for k in range(K):
                H_iid[:, k] /= np.sqrt(self.path_loss[k])
            self.H = H_iid

        elif self.channel_type == 'rician':
            # Rician fading: h_k = sqrt(K_r/(K_r+1)) * h_LOS + sqrt(1/(K_r+1)) * h_NLOS
            K_r = self.rician_factor
            H_los = self._generate_los_component()
            H_nlos = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)

            H_rician = (np.sqrt(K_r / (K_r + 1)) * H_los +
                        np.sqrt(1 / (K_r + 1)) * H_nlos)
            # Ap dung path loss
            for k in range(K):
                H_rician[:, k] /= np.sqrt(self.path_loss[k])
            self.H = H_rician
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

    def _generate_los_component(self):
        """
        Tao thanh phan LOS (Line-of-Sight) dua tren goc den (AoA)
        Steering vector: a(theta) = [1, e^{j*pi*sin(theta)}, ..., e^{j*(M-1)*pi*sin(theta)}]
        """
        M, K = self.M, self.K
        H_los = np.zeros((M, K), dtype=complex)

        for k in range(K):
            # Goc den ngau nhien cho moi user
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            # ULA steering vector
            steering = np.exp(1j * np.pi * np.sin(theta) * np.arange(M))
            H_los[:, k] = steering

        return H_los

    def update_channel(self, correlation=0.9):
        """
        Cap nhat kenh theo thoi gian (time-varying)
        Dung mo hinh Jake's temporal correlation:
            H(t+1) = rho * H(t) + sqrt(1 - rho^2) * H_new

        Args:
            correlation: he so tuong quan thoi gian rho ∈ [0, 1]
        """
        H_old = self.H.copy()
        self.generate_channel()  # Tao kenh moi
        self.H = correlation * H_old + np.sqrt(1 - correlation**2) * self.H

    def get_channel_matrix(self):
        """Tra ve ma tran kenh H ∈ C^{M x K}"""
        return self.H.copy()

    def get_channel_vector(self, k):
        """Tra ve vector kenh h_k ∈ C^{M x 1} cua user k"""
        return self.H[:, k:k+1].copy()

    def get_channel_state_vector(self):
        """
        Chuyen kenh phuc thanh vector thuc de lam state cho DRL
        Tach real va imaginary, flatten

        Returns:
            state_vector: numpy array shape (2*M*K,)
        """
        h_real = self.H.real.flatten()
        h_imag = self.H.imag.flatten()
        return np.concatenate([h_real, h_imag])
