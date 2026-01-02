import math
from typing import Optional, Dict

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """
    Small MLP used inside coupling transforms.
    Output projection is zero-initialized so the coupling starts near identity.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ReversibleCouplingSidecar(nn.Module):
    """
    Invertible additive coupling transform on the last dimension.
    h -> coupling(h), inverse exists in closed form.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 0.25,
        init_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for a 50/50 split, got dim={dim}")
        half = dim // 2
        hidden_dim = max(1, int(round(half * hidden_mult)))

        self.F = TwoLayerMLP(half, hidden_dim, half, dropout=dropout)
        self.G = TwoLayerMLP(half, hidden_dim, half, dropout=dropout)

        # Positive scale via exp(log_scale). Initialize small-ish.
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, h: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        a, b = h.chunk(2, dim=-1)
        s = self.scale()
        if gate is not None:
            # gate shape: [B,1,1], broadcastable to [B,S,1]
            s = s * gate

        a1 = a + s * self.F(b)
        b1 = b + s * self.G(a1)
        return torch.cat([a1, b1], dim=-1)

    @torch.no_grad()
    def inverse(self, h: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        a1, b1 = h.chunk(2, dim=-1)
        s = self.scale()
        if gate is not None:
            s = s * gate

        b = b1 - s * self.G(a1)
        a = a1 - s * self.F(b)
        return torch.cat([a, b], dim=-1)


class SeqGate(nn.Module):
    """
    Scalar gate per sequence, per layer.
    g = sigmoid(w^T mean_t h_t + b)
    """

    def __init__(self, dim: int, init_bias: float = -4.0):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)  # [B,D]
        g = torch.sigmoid(self.proj(pooled))  # [B,1]
        return g.view(-1, 1, 1)  # [B,1,1]


class PatchedGPT2Block(nn.Module):
    """
    Wraps an existing GPT2Block. Calls the base block, then applies an invertible sidecar.
    Preserves the tuple output structure: (hidden_states, present, attentions, ...)
    """

    def __init__(
        self,
        base_block: nn.Module,
        dim: int,
        hidden_mult: float = 0.25,
        init_scale: float = 0.1,
        gate_init_bias: float = -4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base_block
        self.gate = SeqGate(dim, init_bias=gate_init_bias)
        self.sidecar = ReversibleCouplingSidecar(
            dim=dim, hidden_mult=hidden_mult, init_scale=init_scale, dropout=dropout
        )

    def forward(self, *args, **kwargs):
        outputs = self.base(*args, **kwargs)
        u = outputs[0]

        # Gate is computed from the base block output (u), then used to modulate the patch.
        g = self.gate(u)  # [B,1,1]

        # Apply the invertible coupling transform, then turn it into an additive sidecar:
        # h = u + g * (C(u) - u)
        c = self.sidecar(u)  # C(u)
        delta = c - u
        h = u + g * delta

        # Expose last-step tensors for optional regularization/logging in the training loop.
        self._last_gate = g
        self._last_patch_l2 = (g * delta).pow(2).mean()

        return (h,) + outputs[1:]


def patch_gpt2_lm(
    model,
    hidden_mult: float = 0.25,
    init_scale: float = 0.1,
    gate_init_bias: float = -4.0,
):
    """
    Patch GPT2LMHeadModel (or GPT2Model-like) in place.
    Expects blocks in model.transformer.h and embedding dim in model.config.n_embd.
    """
    dim = int(model.config.n_embd)
    for i in range(len(model.transformer.h)):
        model.transformer.h[i] = PatchedGPT2Block(
            base_block=model.transformer.h[i],
            dim=dim,
            hidden_mult=hidden_mult,
            init_scale=init_scale,
            gate_init_bias=gate_init_bias,
        )
    return model


def freeze_all_but_patches(model: nn.Module):
    """
    Freeze everything except parameters that belong to gate or sidecar modules.
    """
    for name, p in model.named_parameters():
        trainable = (".sidecar." in name) or (".gate." in name)
        p.requires_grad = bool(trainable)
    return model


def patch_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only trainable params (patch params) for compact checkpoints.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().cpu()
    return out


def load_patch_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    """
    Load patch params by name. Expects the model already patched.
    """
    name_to_param = dict(model.named_parameters())
    missing = []
    for name, t in state.items():
        if name not in name_to_param:
            missing.append(name)
            continue
        name_to_param[name].data.copy_(
            t.to(name_to_param[name].device, dtype=name_to_param[name].dtype)
        )
    if missing:
        raise KeyError(
            f"Missing {len(missing)} patch params in current model (first few): {missing[:5]}"
        )
