import torch
import torch.nn as nn
import math


class PhaseAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def normalize_to_unit_phase(self, K, Q):
        return K / (K.norm(dim=-1, keepdim=True) + 1e-8), Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)

    def phase_difference(self, K, Q):
        phase_diff = torch.einsum("bqd,bkd->bqk", Q, K)  # (B, S, S)
        phase = torch.cos(phase_diff * math.pi)  # Interference term
        return phase

    def compute_amplitudes(self, K, Q, phase):
        amplitude = torch.einsum("bqd,bkd->bqk", Q, K) * self.scale
        return torch.softmax(phase * amplitude, dim=-1)

    def apply_attention(self, V, weight):
        out = torch.einsum("bqk,bkd->bqd", weight, V)
        out = self.out_proj(out)
        return out

    def qkv_projection(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        Q, K, V = self.qkv_projection(x)  # (B, S, D)

        phase = self.phase_difference(*self.normalize_to_unit_phase(K, Q))
        weight = self.compute_amplitudes(K, Q, phase)
        return self.apply_attention(V, weight)
