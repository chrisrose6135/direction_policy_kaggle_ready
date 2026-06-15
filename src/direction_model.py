from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


DIRECTION_CLASS_NAMES = {0: 'SELL', 1: 'NO_TRADE', 2: 'BUY'}
DIRECTION_CLASS_IDS = {'SELL': 0, 'NO_TRADE': 1, 'BUY': 2}


SUPPORTED_ARCHITECTURES = {
    'residual_mlp_gate_direction_v1',
    'hierarchical_tcn_edge_v1',
    'small_transformer_gate_direction_v1',
    'inception_time_gate_direction_v1',
    'mixture_of_experts_direction_v1',
}

_ARCHITECTURE_ALIASES = {
    'tcn': 'hierarchical_tcn_edge_v1',
    'hierarchical_tcn': 'hierarchical_tcn_edge_v1',
    'hierarchical_tcn_gate_direction_edge': 'hierarchical_tcn_edge_v1',
    'mlp': 'residual_mlp_gate_direction_v1',
    'residual_mlp': 'residual_mlp_gate_direction_v1',
    'transformer': 'small_transformer_gate_direction_v1',
    'small_transformer': 'small_transformer_gate_direction_v1',
    'inception': 'inception_time_gate_direction_v1',
    'inception_time': 'inception_time_gate_direction_v1',
    'moe': 'mixture_of_experts_direction_v1',
    'mixture_of_experts': 'mixture_of_experts_direction_v1',
}


def _canonical_architecture(value: Any) -> str:
    raw = str(value or 'hierarchical_tcn_edge_v1').strip()
    key = raw.lower()
    out = _ARCHITECTURE_ALIASES.get(key, raw)
    if out not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported model.architecture={raw!r}. Supported values: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    return out


def _as_positive_int(value: Any, default: int, *, name: str) -> int:
    if value is None:
        value = default
    out = int(value)
    if out < 1:
        raise ValueError(f'model.{name} must be >= 1, got {out}')
    return out


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _int_list(value: Any, default: list[int], *, name: str) -> list[int]:
    if value is None:
        values = list(default)
    elif isinstance(value, int):
        values = [int(value)]
    elif isinstance(value, str):
        values = [int(part.strip()) for part in value.split(',') if part.strip()]
    else:
        values = [int(v) for v in value]
    if not values:
        raise ValueError(f'model.{name} must contain at least one value')
    if any(v < 1 for v in values):
        raise ValueError(f'model.{name} values must all be >= 1, got {values}')
    return values


def _str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    return [str(v) for v in value]


def _expand_or_trim(values: list[int], n: int) -> list[int]:
    if len(values) >= n:
        return values[:n]
    return values + [values[-1]] * (n - len(values))


def _feature_columns_from_cfg(cfg: dict[str, Any]) -> list[str]:
    if isinstance(cfg.get('_feature_columns'), list):
        return [str(x) for x in cfg['_feature_columns']]
    mcfg = cfg.get('model', {}) or {}
    if isinstance(mcfg.get('feature_columns'), list):
        return [str(x) for x in mcfg['feature_columns']]
    fcfg = cfg.get('features', {}) or {}
    if isinstance(fcfg.get('include_columns'), list):
        return [str(x) for x in fcfg['include_columns']]
    return []


def direction_probabilities_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return SELL/NO_TRADE/BUY probabilities from model outputs."""
    if 'direction_probabilities' in outputs:
        return outputs['direction_probabilities']
    return torch.softmax(outputs['direction_logits'], dim=-1)


class ResidualTemporalBlock(nn.Module):
    """A compact residual dilated Conv1D block for fixed-window price sequences."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f'TCN kernel sizes should be odd so sequence length is preserved; got {kernel_size}')
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualDenseBlock(nn.Module):
    """Small residual MLP block used by the tabular MLP and MoE experts."""

    def __init__(self, features: int, hidden: int | None = None, dropout: float = 0.0, *, norm: bool = True):
        super().__init__()
        hidden = int(hidden or features)
        layers: list[nn.Module] = [nn.Linear(features, hidden)]
        if norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, features))
        if norm:
            layers.append(nn.LayerNorm(features))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MLPStack(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        *,
        dropout: float = 0.0,
        residual_blocks: int = 0,
        norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(input_size)
        for hidden in hidden_sizes:
            hidden = int(hidden)
            layers.append(nn.Linear(in_features, hidden))
            if norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        for _ in range(int(residual_blocks or 0)):
            layers.append(ResidualDenseBlock(in_features, dropout=dropout, norm=norm))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_size = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNEncoder(nn.Module):
    def __init__(self, n_features: int, mcfg: dict[str, Any], *, dropout: float):
        super().__init__()
        tcn_dropout = float(mcfg.get('tcn_dropout', mcfg.get('conv_dropout', dropout)) or 0.0)
        dense_dropout = float(mcfg.get('dense_dropout', dropout) or 0.0)
        channels = _as_positive_int(mcfg.get('tcn_channels', mcfg.get('conv_channels')), 64, name='tcn_channels')
        tcn_blocks = _as_positive_int(mcfg.get('tcn_blocks'), 4, name='tcn_blocks')
        kernel_size = _as_positive_int(mcfg.get('tcn_kernel_size', mcfg.get('conv_kernel_size')), 5, name='tcn_kernel_size')
        if kernel_size % 2 == 0:
            raise ValueError(f'model.tcn_kernel_size should be odd, got {kernel_size}')
        dilations = _expand_or_trim(_int_list(mcfg.get('tcn_dilations'), [1, 2, 4, 8], name='tcn_dilations'), tcn_blocks)
        self.input_projection = nn.Sequential(
            nn.Conv1d(int(n_features), channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )
        self.tcn = nn.Sequential(*[
            ResidualTemporalBlock(channels, kernel_size=kernel_size, dilation=int(d), dropout=tcn_dropout)
            for d in dilations
        ])
        pooled_features = channels * 3
        default_dense_hidden = int(mcfg.get('dense_hidden_size', 64))
        dense_hidden_sizes = _int_list(
            mcfg.get('dense_hidden_sizes'),
            [default_dense_hidden, max(16, default_dense_hidden // 2)],
            name='dense_hidden_sizes',
        )
        if mcfg.get('dense_layers') is not None:
            dense_layers = _as_positive_int(mcfg.get('dense_layers'), len(dense_hidden_sizes), name='dense_layers')
            dense_hidden_sizes = _expand_or_trim(dense_hidden_sizes, dense_layers)
        self.projection = MLPStack(pooled_features, dense_hidden_sizes, dropout=dense_dropout, residual_blocks=0, norm=True)
        self.output_size = self.projection.output_size
        self.details = {
            'encoder': 'tcn',
            'tcn_channels': int(channels),
            'tcn_blocks': int(tcn_blocks),
            'tcn_kernel_size': int(kernel_size),
            'tcn_dilations': [int(d) for d in dilations],
            'tcn_receptive_field_bars': int(1 + 2 * (kernel_size - 1) * sum(int(d) for d in dilations)),
            'temporal_pooling': ['last', 'mean', 'max'],
            'dense_hidden_sizes': [int(v) for v in dense_hidden_sizes],
            'tcn_dropout': float(tcn_dropout),
            'dense_dropout': float(dense_dropout),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.input_projection(z)
        z = self.tcn(z)
        rep = torch.cat([z[:, :, -1], z.mean(dim=-1), z.amax(dim=-1)], dim=1)
        return self.projection(rep)


class ResidualMLPEncoder(nn.Module):
    def __init__(self, n_features: int, mcfg: dict[str, Any], *, dropout: float):
        super().__init__()
        mode = str(mcfg.get('mlp_input_mode', 'last')).lower()
        if mode not in {'last', 'mean', 'last_mean', 'last_mean_max', 'flatten'}:
            raise ValueError('model.mlp_input_mode must be one of: last, mean, last_mean, last_mean_max, flatten')
        seq_len = _as_positive_int(mcfg.get('sequence_length'), 64, name='sequence_length')
        if mode == 'last':
            input_size = n_features
        elif mode == 'mean':
            input_size = n_features
        elif mode == 'last_mean':
            input_size = n_features * 2
        elif mode == 'last_mean_max':
            input_size = n_features * 3
        else:
            input_size = n_features * seq_len
        hidden = _int_list(mcfg.get('mlp_hidden_sizes', mcfg.get('dense_hidden_sizes')), [256, 128, 64], name='mlp_hidden_sizes')
        residual_blocks = int(mcfg.get('mlp_residual_blocks', 2) or 0)
        self.mode = mode
        self.seq_len = seq_len
        self.stack = MLPStack(input_size, hidden, dropout=dropout, residual_blocks=residual_blocks, norm=True)
        self.output_size = self.stack.output_size
        self.details = {
            'encoder': 'residual_mlp',
            'mlp_input_mode': mode,
            'mlp_hidden_sizes': [int(v) for v in hidden],
            'mlp_residual_blocks': int(residual_blocks),
        }

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'last':
            return x[:, -1, :]
        if self.mode == 'mean':
            return x.mean(dim=1)
        if self.mode == 'last_mean':
            return torch.cat([x[:, -1, :], x.mean(dim=1)], dim=-1)
        if self.mode == 'last_mean_max':
            return torch.cat([x[:, -1, :], x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(self._select(x))


class SmallTransformerEncoder(nn.Module):
    def __init__(self, n_features: int, mcfg: dict[str, Any], *, dropout: float):
        super().__init__()
        seq_len = _as_positive_int(mcfg.get('sequence_length'), 64, name='sequence_length')
        d_model = _as_positive_int(mcfg.get('d_model', mcfg.get('transformer_d_model')), 64, name='d_model')
        n_heads = _as_positive_int(mcfg.get('n_heads', mcfg.get('transformer_heads')), 4, name='n_heads')
        if d_model % n_heads != 0:
            raise ValueError(f'model.d_model ({d_model}) must be divisible by model.n_heads ({n_heads})')
        num_layers = _as_positive_int(mcfg.get('num_layers', mcfg.get('transformer_layers')), 2, name='num_layers')
        ff_dim = _as_positive_int(mcfg.get('feedforward_dim', mcfg.get('transformer_feedforward_dim')), 128, name='feedforward_dim')
        pooling = str(mcfg.get('pooling', 'attention')).lower()
        if pooling not in {'last', 'mean', 'last_mean', 'attention'}:
            raise ValueError('model.pooling for transformer must be one of: last, mean, last_mean, attention')
        self.pooling = pooling
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(d_model, 1) if pooling == 'attention' else None
        pooled = d_model * 2 if pooling == 'last_mean' else d_model
        dense_hidden_sizes = _int_list(mcfg.get('dense_hidden_sizes'), [d_model, max(16, d_model // 2)], name='dense_hidden_sizes')
        dense_dropout = float(mcfg.get('dense_dropout', dropout) or 0.0)
        self.projection = MLPStack(pooled, dense_hidden_sizes, dropout=dense_dropout, residual_blocks=0, norm=True)
        self.output_size = self.projection.output_size
        self.details = {
            'encoder': 'small_transformer',
            'sequence_length': int(seq_len),
            'd_model': int(d_model),
            'n_heads': int(n_heads),
            'num_layers': int(num_layers),
            'feedforward_dim': int(ff_dim),
            'pooling': pooling,
            'dense_hidden_sizes': [int(v) for v in dense_hidden_sizes],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        z = self.input_projection(x) + self.pos_embedding[:, :t, :]
        z = self.encoder(z)
        if self.pooling == 'last':
            pooled = z[:, -1, :]
        elif self.pooling == 'mean':
            pooled = z.mean(dim=1)
        elif self.pooling == 'last_mean':
            pooled = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=-1)
        else:
            weights = torch.softmax(self.attention_pool(z).squeeze(-1), dim=1)
            pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.projection(pooled)


class InceptionTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        kernels = [int(k) for k in kernel_sizes]
        if any(k % 2 == 0 for k in kernels):
            raise ValueError(f'InceptionTime kernel sizes should be odd, got {kernels}')
        branch_count = len(kernels) + 1  # conv branches + maxpool branch
        branch_channels = max(1, out_channels // branch_count)
        actual_out = branch_channels * branch_count
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(1, actual_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Identity() if in_channels == actual_out else nn.Conv1d(in_channels, actual_out, kernel_size=1)
        self.output_channels = actual_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [conv(x) for conv in self.conv_branches]
        branches.append(self.pool_branch(x))
        z = torch.cat(branches, dim=1)
        z = self.dropout(self.act(self.norm(z)))
        return self.act(z + self.residual(x))


class InceptionTimeEncoder(nn.Module):
    def __init__(self, n_features: int, mcfg: dict[str, Any], *, dropout: float):
        super().__init__()
        channels = _as_positive_int(mcfg.get('channels', mcfg.get('inception_channels')), 32, name='channels')
        blocks = _as_positive_int(mcfg.get('blocks', mcfg.get('inception_blocks')), 3, name='blocks')
        kernels = _int_list(mcfg.get('kernel_sizes', mcfg.get('inception_kernel_sizes')), [3, 5, 9, 17], name='kernel_sizes')
        inception_dropout = float(mcfg.get('inception_dropout', dropout) or 0.0)
        layers: list[nn.Module] = []
        in_ch = int(n_features)
        for _ in range(blocks):
            block = InceptionTimeBlock(in_ch, channels, kernels, dropout=inception_dropout)
            layers.append(block)
            in_ch = block.output_channels
        self.net = nn.Sequential(*layers)
        pooled = in_ch * 3
        dense_hidden_sizes = _int_list(mcfg.get('dense_hidden_sizes'), [64, 32], name='dense_hidden_sizes')
        dense_dropout = float(mcfg.get('dense_dropout', dropout) or 0.0)
        self.projection = MLPStack(pooled, dense_hidden_sizes, dropout=dense_dropout, residual_blocks=0, norm=True)
        self.output_size = self.projection.output_size
        self.details = {
            'encoder': 'inception_time',
            'inception_channels_requested': int(channels),
            'inception_channels_actual': int(in_ch),
            'inception_blocks': int(blocks),
            'inception_kernel_sizes': [int(v) for v in kernels],
            'inception_dropout': float(inception_dropout),
            'temporal_pooling': ['last', 'mean', 'max'],
            'dense_hidden_sizes': [int(v) for v in dense_hidden_sizes],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        pooled = torch.cat([z[:, :, -1], z.mean(dim=-1), z.amax(dim=-1)], dim=1)
        return self.projection(pooled)


class MixtureOfExpertsEncoder(nn.Module):
    def __init__(self, n_features: int, cfg: dict[str, Any], *, dropout: float):
        super().__init__()
        mcfg = cfg.get('model', {}) or {}
        self.feature_columns = _feature_columns_from_cfg(cfg)
        self.n_features = int(n_features)
        self.num_experts = _as_positive_int(mcfg.get('num_experts'), 4, name='num_experts')
        expert_hidden = _as_positive_int(mcfg.get('expert_hidden_size'), 64, name='expert_hidden_size')
        expert_layers = _as_positive_int(mcfg.get('expert_layers'), 2, name='expert_layers')
        self.expert_input_mode = str(mcfg.get('expert_input_mode', 'last_mean')).lower()
        if self.expert_input_mode not in {'last', 'last_mean', 'last_mean_max'}:
            raise ValueError('model.expert_input_mode must be one of: last, last_mean, last_mean_max')
        if self.expert_input_mode == 'last':
            expert_input_size = n_features
        elif self.expert_input_mode == 'last_mean':
            expert_input_size = n_features * 2
        else:
            expert_input_size = n_features * 3
        expert_hidden_sizes = [expert_hidden] * expert_layers
        self.experts = nn.ModuleList([
            MLPStack(expert_input_size, expert_hidden_sizes, dropout=dropout, residual_blocks=1, norm=True)
            for _ in range(self.num_experts)
        ])
        self.expert_output_size = self.experts[0].output_size

        router_inputs = _str_list(mcfg.get('router_inputs'), [
            'sig_analytic_signal_class',
            'sig_buy_signal_count',
            'sig_sell_signal_count',
            'sig_signal_conflict',
            'sig_net_signal_vote',
            'sig_adx_strength',
            'sig_atr_zscore',
        ])
        index_map = {name: i for i, name in enumerate(self.feature_columns)}
        self.router_input_names = [name for name in router_inputs if name in index_map]
        self.router_indices = [index_map[name] for name in self.router_input_names]
        self.router_use_full_last_fallback = len(self.router_indices) == 0
        router_input_size = len(self.router_indices) if self.router_indices else n_features
        router_hidden = _as_positive_int(mcfg.get('router_hidden_size'), 32, name='router_hidden_size')
        self.router = nn.Sequential(
            nn.Linear(router_input_size, router_hidden),
            nn.LayerNorm(router_hidden),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(router_hidden, self.num_experts),
        )
        final_hidden = _int_list(mcfg.get('dense_hidden_sizes'), [self.expert_output_size], name='dense_hidden_sizes')
        self.projection = MLPStack(self.expert_output_size, final_hidden, dropout=dropout, residual_blocks=0, norm=True)
        self.output_size = self.projection.output_size
        self.details = {
            'encoder': 'mixture_of_experts',
            'num_experts': int(self.num_experts),
            'expert_hidden_size': int(expert_hidden),
            'expert_layers': int(expert_layers),
            'expert_input_mode': self.expert_input_mode,
            'router_hidden_size': int(router_hidden),
            'router_inputs_configured': router_inputs,
            'router_inputs_used': self.router_input_names,
            'router_uses_full_last_fallback': bool(self.router_use_full_last_fallback),
            'dense_hidden_sizes': [int(v) for v in final_hidden],
        }

    def _expert_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.expert_input_mode == 'last':
            return x[:, -1, :]
        if self.expert_input_mode == 'last_mean':
            return torch.cat([x[:, -1, :], x.mean(dim=1)], dim=-1)
        return torch.cat([x[:, -1, :], x.mean(dim=1), x.amax(dim=1)], dim=-1)

    def _router_input(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        if self.router_indices:
            idx = torch.as_tensor(self.router_indices, dtype=torch.long, device=x.device)
            return last.index_select(dim=1, index=idx)
        return last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_in = self._expert_input(x)
        expert_reps = torch.stack([expert(expert_in) for expert in self.experts], dim=1)  # B,E,H
        weights = torch.softmax(self.router(self._router_input(x)), dim=-1)
        mixed = torch.sum(expert_reps * weights.unsqueeze(-1), dim=1)
        return self.projection(mixed)


class DirectionTradePolicyNet(nn.Module):
    """Hierarchical direction policy network with selectable encoders.

    Select the encoder with ``model.architecture`` in the config. All supported
    architectures keep the same output contract used by training, replay and
    live/demo trading:

        - TRADE / NO_TRADE gate
        - SELL / BUY side head conditional on a trade
        - combined SELL / NO_TRADE / BUY probabilities
        - optional BUY/SELL edge-pips head
        - optional analytic-signal auxiliary head
    """

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        mcfg = cfg.get('model', {}) or {}
        lcfg = cfg.get('labels', {}) or {}

        self.architecture = _canonical_architecture(mcfg.get('architecture', 'hierarchical_tcn_edge_v1'))
        self.is_hierarchical = True
        self.n_features = int(n_features)

        dropout = float(mcfg.get('dropout', 0.0) or 0.0)
        if self.architecture == 'hierarchical_tcn_edge_v1':
            self.encoder = TCNEncoder(n_features, mcfg, dropout=dropout)
        elif self.architecture == 'residual_mlp_gate_direction_v1':
            self.encoder = ResidualMLPEncoder(n_features, mcfg, dropout=dropout)
        elif self.architecture == 'small_transformer_gate_direction_v1':
            self.encoder = SmallTransformerEncoder(n_features, mcfg, dropout=dropout)
        elif self.architecture == 'inception_time_gate_direction_v1':
            self.encoder = InceptionTimeEncoder(n_features, mcfg, dropout=dropout)
        elif self.architecture == 'mixture_of_experts_direction_v1':
            self.encoder = MixtureOfExpertsEncoder(n_features, cfg, dropout=dropout)
        else:  # defensive; canonicalisation should already catch this.
            raise ValueError(f'Unsupported model architecture: {self.architecture}')

        self.representation_size = int(self.encoder.output_size)
        self.trade_gate_head = nn.Linear(self.representation_size, 1)
        self.side_direction_head = nn.Linear(self.representation_size, 2)  # SELL, BUY conditional on trade
        self.use_edge_pips_head = _as_bool(mcfg.get('use_edge_pips_head'), True)
        self.edge_pips_scale = float(mcfg.get('edge_pips_scale', lcfg.get('take_profit_pips', 10.0)) or 10.0)
        if self.edge_pips_scale <= 0:
            self.edge_pips_scale = 10.0
        self.edge_head = nn.Linear(self.representation_size, 2) if self.use_edge_pips_head else None
        self.use_analytic_signal_agreement_head = _as_bool(mcfg.get('use_analytic_signal_agreement_head'), False)
        self.analytic_signal_agreement_head = (
            nn.Linear(self.representation_size, 3) if self.use_analytic_signal_agreement_head else None
        )

        model_type = f'{self.architecture}_hierarchical_gate_direction'
        encoder_details = getattr(self.encoder, 'details', {})
        self.model_details = {
            'architecture': self.architecture,
            'model_type': model_type,
            'supported_architectures': sorted(SUPPORTED_ARCHITECTURES),
            'output_order': {
                'gate': 'TRADE vs NO_TRADE',
                'side_direction': {0: 'SELL_GIVEN_TRADE', 1: 'BUY_GIVEN_TRADE'},
                'combined_direction': DIRECTION_CLASS_NAMES,
                'edge_pips': {0: 'buy_edge_pips', 1: 'sell_edge_pips'} if self.use_edge_pips_head else None,
                'analytic_signal_agreement': DIRECTION_CLASS_NAMES if self.use_analytic_signal_agreement_head else None,
            },
            **encoder_details,
            'dropout': float(dropout),
            'representation_size': int(self.representation_size),
            'is_hierarchical': True,
            'use_edge_pips_head': bool(self.use_edge_pips_head),
            'edge_pips_scale': float(self.edge_pips_scale),
            'use_analytic_signal_agreement_head': bool(self.use_analytic_signal_agreement_head),
        }

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rep = self._encode(x)
        trade_logit = self.trade_gate_head(rep).squeeze(-1)
        side_logits = self.side_direction_head(rep)
        trade_prob = torch.sigmoid(trade_logit)
        side_probs = torch.softmax(side_logits, dim=-1)

        sell_prob = trade_prob * side_probs[:, 0]
        buy_prob = trade_prob * side_probs[:, 1]
        no_trade_prob = 1.0 - trade_prob
        probs = torch.stack([sell_prob, no_trade_prob, buy_prob], dim=1)
        direction_logits = torch.log(torch.clamp(probs, min=1e-7, max=1.0))

        out: dict[str, torch.Tensor] = {
            'direction_logits': direction_logits,
            'direction_probabilities': probs,
            'trade_logit': trade_logit,
            'trade_probability': trade_prob,
            'no_trade_probability': no_trade_prob,
            'side_direction_logits': side_logits,
            'side_sell_probability': side_probs[:, 0],
            'side_buy_probability': side_probs[:, 1],
            'sell_probability': sell_prob,
            'buy_probability': buy_prob,
        }
        if self.edge_head is not None:
            edge_norm = self.edge_head(rep)
            edge_pips = edge_norm * float(self.edge_pips_scale)
            out['edge_pips_normalized'] = edge_norm
            out['edge_pips'] = edge_pips
            out['buy_edge_pips'] = edge_pips[:, 0]
            out['sell_edge_pips'] = edge_pips[:, 1]
        if self.analytic_signal_agreement_head is not None:
            out['analytic_signal_logits'] = self.analytic_signal_agreement_head(rep)
        return out
