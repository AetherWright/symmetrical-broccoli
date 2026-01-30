import asyncio
import atexit
import codecs
import functools
import gc
import json
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn
from contextlib import nullcontext

ALLOWED_SEGMENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.")
STORAGE_ROOT = os.environ.get("TF_SERVER_STORAGE_ROOT") or os.path.join(os.getcwd(), "tf_server_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)

LOGGER = logging.getLogger("tf_server")


def _configure_torch_devices() -> Dict[str, Any]:
    gpu_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if gpu_available else 0
    if not gpu_available:
        LOGGER.info("No CUDA devices detected; running on CPU.")
        return {
            "available": False,
            "count": 0,
            "names": [],
        }
    names = []
    for index in range(device_count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except Exception as exc:  # pragma: no cover - defensive
            names.append(f"cuda:{index} ({exc})")
    LOGGER.info("CUDA acceleration enabled (%d device(s)).", device_count)
    return {"available": True, "count": device_count, "names": names}


def _configure_float_policy(gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    requested_gpu = str(os.environ.get("TF_SERVER_ENABLE_GPU", "0")).lower() in {"1", "true", "yes", "on"}
    use_gpu = bool(requested_gpu and gpu_info.get("available"))
    if use_gpu:
        device = torch.device("cuda")
        torch.cuda.init()
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    LOGGER.info("Torch default dtype set to %s on device %s", dtype, device)
    return {
        "device": device,
        "dtype": dtype,
        "using_gpu": use_gpu,
    }


def _sanitize_segment(value: Any, fallback: str = "default") -> str:
    if not value:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    sanitized = "".join(ch for ch in text if ch in ALLOWED_SEGMENT_CHARS)
    return sanitized or fallback


def _resolve_storage_path(*segments: str, create: bool = False) -> str:
    parts = [_sanitize_segment(seg) for seg in segments if seg is not None]
    directory = os.path.join(STORAGE_ROOT, *parts)
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory


app = FastAPI()

BRAINS: Dict[str, "RemoteBrain"] = {}
STATE_LOCK = threading.Lock()


STATUS: Dict[str, Any] = {
    "started_at": time.time(),
    "total_requests": 0,
    "requests": [],
    "bots": {},
    "workers": {},
}

GPU_INFO = _configure_torch_devices()
STATUS["accelerators"] = {"gpu": GPU_INFO}
FLOAT_POLICY = _configure_float_policy(GPU_INFO)
STATUS["float_policy"] = {
    "dtype": str(FLOAT_POLICY["dtype"]),
    "using_gpu": FLOAT_POLICY["using_gpu"],
    "device": str(FLOAT_POLICY["device"]),
}


class MemoryManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_collect = 0.0
        self.interval = max(0.0, float(os.environ.get("TF_SERVER_GC_INTERVAL", 0.0)))

    def maybe_collect(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self.interval > 0.0 and now - self._last_collect < self.interval:
            return {"performed": False}
        with self._lock:
            now = time.time()
            if not force and self.interval > 0.0 and now - self._last_collect < self.interval:
                return {"performed": False}
            collected = gc.collect()
            gpu_reclaimed = False
            if FLOAT_POLICY.get("using_gpu"):
                try:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    gpu_reclaimed = True
                except Exception:  # pragma: no cover - defensive cleanup
                    LOGGER.exception("Failed to release CUDA caches during GC")
            self._last_collect = now
            return {"performed": True, "objects_collected": int(collected), "gpu_caches_cleared": gpu_reclaimed}


MEMORY_MANAGER = MemoryManager()


class EndpointWorkerPool:
    def __init__(self, limits: Dict[str, int]):
        self._limits = dict(limits)
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.Lock()

    def _get_limit(self, endpoint: str) -> int:
        return max(1, int(self._limits.get(endpoint, self._limits.get("default", 1))))

    def _get_executor(self, endpoint: str) -> ThreadPoolExecutor:
        with self._lock:
            executor = self._executors.get(endpoint)
            if executor is None:
                max_workers = self._get_limit(endpoint)
                executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{endpoint}-worker")
                self._executors[endpoint] = executor
            return executor

    def run(self, endpoint: str, func, *args, **kwargs):
        executor = self._get_executor(endpoint)
        future = executor.submit(func, *args, **kwargs)
        return future.result()

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        snapshot: Dict[str, Dict[str, int]] = {}
        with self._lock:
            for endpoint, executor in self._executors.items():
                queue = getattr(executor, "_work_queue", None)
                pending = queue.qsize() if queue is not None else 0
                snapshot[endpoint] = {
                    "max_workers": executor._max_workers,  # type: ignore[attr-defined]
                    "pending": pending,
                }
        return snapshot

    def shutdown(self) -> None:
        with self._lock:
            executors = list(self._executors.values())
            self._executors.clear()
        for executor in executors:
            executor.shutdown(wait=False)


def _read_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


DEFAULT_WORKERS = _read_int("TF_SERVER_DEFAULT_WORKERS", 2)
ENDPOINT_WORKER_LIMITS = {
    "act": _read_int("TF_SERVER_ACT_WORKERS", DEFAULT_WORKERS * 2),
    "train": _read_int("TF_SERVER_TRAIN_WORKERS", DEFAULT_WORKERS),
    "average": _read_int("TF_SERVER_AVERAGE_WORKERS", DEFAULT_WORKERS),
    "mutate": _read_int("TF_SERVER_MUTATE_WORKERS", DEFAULT_WORKERS),
    "copy": _read_int("TF_SERVER_COPY_WORKERS", DEFAULT_WORKERS),
    "save": _read_int("TF_SERVER_SAVE_WORKERS", DEFAULT_WORKERS),
    "load": _read_int("TF_SERVER_LOAD_WORKERS", DEFAULT_WORKERS),
    "default": DEFAULT_WORKERS,
}

WORKERS = EndpointWorkerPool(ENDPOINT_WORKER_LIMITS)
atexit.register(WORKERS.shutdown)

BRAIN_CONFIG = {
    "storsky_units": [128, 96],
    "dropout_rate": 0.2,
    "memory_decay": 0.9,
    "memory_scale": 0.15,
    "action_branches": [8, 4, 4, 1, 3, 1, 5, 8],
    "transformer_tokens": 4,
    "transformer_embed": 24,
    "transformer_heads": 3,
    "transformer_layers": 1,
    "transformer_ff": 96,
    "transformer_rotary_base": 10000.0,
    "learning_rate": 2e-3,
    "lion_beta_1": 0.9,
    "lion_beta_2": 0.99,
    "lion_weight_decay": 0.0,
    "lookahead_sync": 6,
    "lookahead_alpha": 0.5,
    "ema_decay": 0.995,
    "reward_prediction_weight": 0.35,
    "bot_heads": 4,
}

OBS_CLAMP = abs(_read_float("TF_SERVER_OBSERVATION_CLAMP", 1e6))
GRAD_CLIP_VALUE = abs(_read_float("TF_SERVER_GRADIENT_CLIP_VALUE", 100.0))
GRAD_CLIP_GLOBAL_NORM = abs(_read_float("TF_SERVER_GRADIENT_GLOBAL_NORM", 250.0))
GRAD_SKIP_GLOBAL_NORM = abs(_read_float("TF_SERVER_GRADIENT_SKIP_GLOBAL_NORM", 0.0))
WEIGHT_CLAMP = abs(_read_float("TF_SERVER_WEIGHT_CLAMP", 1e6))
ACT_SANITIZATION_STRIKE_WINDOW = max(1.0, _read_float("TF_SERVER_ACT_STRIKE_WINDOW", 30.0))
ACT_SANITIZATION_STRIKE_THRESHOLD = max(1, _read_int("TF_SERVER_ACT_STRIKE_THRESHOLD", 2))
ACT_SANITIZATION_FORCE_THRESHOLD = max(
    ACT_SANITIZATION_STRIKE_THRESHOLD,
    _read_int("TF_SERVER_ACT_FORCE_THRESHOLD", 4),
)


NP_FLOAT = np.float32 if FLOAT_POLICY["dtype"] == torch.float32 else np.float64


def natural_log_relu(inputs: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.relu(inputs))


class LookaheadOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, sync_period: int = 6, slow_step_size: float = 0.5):
        self.optimizer = optimizer
        self.sync_period = max(1, int(sync_period))
        self.slow_step_size = float(slow_step_size)
        self._fast_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        self._slow_params = [p.detach().clone().to(p.device) for p in self._fast_params]
        self._step = 0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _maybe_refresh(self) -> None:
        if len(self._slow_params) != len(self._fast_params):
            self._fast_params = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
            self._slow_params = [p.detach().clone().to(p.device) for p in self._fast_params]

    def step(self, grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None) -> None:
        self._maybe_refresh()
        if grad_scaler is not None and getattr(grad_scaler, "is_enabled", lambda: False)():
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            self.optimizer.step()
        self._step += 1
        if self.sync_period and self._step % self.sync_period == 0:
            for fast, slow in zip(self._fast_params, self._slow_params):
                slow.add_(self.slow_step_size * (fast.data - slow))
                fast.data.copy_(slow)

    def sync_slow_parameters(self, params: Optional[List[torch.nn.Parameter]] = None) -> None:
        self._maybe_refresh()
        if params is None:
            params = self._fast_params
        param_ids = {id(p): index for index, p in enumerate(self._fast_params)}
        for param in params:
            index = param_ids.get(id(param))
            if index is None:
                continue
            self._slow_params[index].data.copy_(param.data)


class StorskyLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.norm = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.use_residual = in_features == out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.norm(self.linear(inputs)))
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


class ParameterEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(min(max(decay, 0.0), 0.999999))
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.sync(model)

    def sync(self, model: torch.nn.Module) -> None:
        self.shadow = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        if not self.shadow:
            self.sync(model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            current = param.detach()
            cached = self.shadow.get(name)
            if cached is None or cached.shape != current.shape:
                self.shadow[name] = current.clone()
            else:
                cached.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        if not self.shadow:
            self.sync(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.detach().clone()
            shadow = self.shadow.get(name)
            if shadow is not None:
                param.data.copy_(shadow)

    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            backup = self.backup.get(name)
            if backup is not None:
                param.data.copy_(backup)
        self.backup = {}


def _apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = tensor[..., ::2]
    odd = tensor[..., 1::2]
    cos = cos[..., :even.shape[-1]]
    sin = sin[..., :even.shape[-1]]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    output = torch.empty_like(tensor)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin


class RotarySelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, base: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Embed dimension must be divisible by number of heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for rotary embeddings")
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, base)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cos, sin = self.rotary(seq_len, inputs.device, inputs.dtype)
        q = _apply_rotary_pos_emb(q, cos, sin)
        k = _apply_rotary_pos_emb(k, cos, sin)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class RotaryEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        rotary_base: float,
    ):
        super().__init__()
        self.self_attn = RotarySelfAttention(embed_dim, num_heads, dropout, rotary_base)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.linear1 = torch.nn.Linear(embed_dim, dim_feedforward)
        self.activation = torch.nn.GELU() if activation == "gelu" else torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, embed_dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inputs)
        attn_output = self.self_attn(x)
        inputs = inputs + self.dropout1(attn_output)
        x = self.norm2(inputs)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return inputs + self.dropout2(ff_output)


class RotaryTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        rotary_base: float,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                RotaryEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, activation, rotary_base)
                for _ in range(max(1, num_layers))
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


class BrainModel(torch.nn.Module):
    def __init__(self, input_size: int, action_count: int):
        super().__init__()
        self.input_bn = torch.nn.BatchNorm1d(input_size)
        storsky_units = BRAIN_CONFIG.get("storsky_units", [128, 96])
        if len(storsky_units) < 2:
            raise ValueError("storsky_units must define two layer sizes")
        self.storsky_1 = StorskyLayer(input_size, int(storsky_units[0]), BRAIN_CONFIG["dropout_rate"])
        self.storsky_2 = StorskyLayer(int(storsky_units[0]), int(storsky_units[1]), BRAIN_CONFIG["dropout_rate"])
        self.memory_dim = int(storsky_units[1])
        self.memory_decay = float(BRAIN_CONFIG.get("memory_decay", 0.9))
        self.memory_scale = float(BRAIN_CONFIG.get("memory_scale", 0.15))
        self.memory_norm = torch.nn.LayerNorm(self.memory_dim)
        self.memory_project = torch.nn.Linear(self.memory_dim, self.memory_dim)
        self.transformer_tokens = int(BRAIN_CONFIG["transformer_tokens"])
        self.transformer_embed = int(BRAIN_CONFIG["transformer_embed"])
        total_transformer_dim = self.transformer_tokens * self.transformer_embed
        if total_transformer_dim <= 0:
            raise ValueError("Transformer configuration must produce a positive feature size")
        if self.transformer_embed % max(1, int(BRAIN_CONFIG["transformer_heads"])) != 0:
            raise ValueError("Transformer embed dimension must be divisible by the number of heads")
        self.transformer_project = torch.nn.Linear(int(storsky_units[1]), total_transformer_dim)
        self.transformer_encoder = RotaryTransformerEncoder(
            num_layers=int(BRAIN_CONFIG["transformer_layers"]),
            embed_dim=self.transformer_embed,
            num_heads=int(BRAIN_CONFIG["transformer_heads"]),
            dim_feedforward=int(BRAIN_CONFIG["transformer_ff"]),
            dropout=BRAIN_CONFIG["dropout_rate"],
            activation="gelu",
            rotary_base=float(BRAIN_CONFIG["transformer_rotary_base"]),
        )
        self.transformer_input_norm = torch.nn.LayerNorm(self.transformer_embed)
        self.transformer_output_norm = torch.nn.LayerNorm(self.transformer_embed)
        self.transformer_dropout = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.transformer_dense = torch.nn.Linear(self.transformer_embed, self.transformer_embed)
        self.action_count = action_count
        branch_sizes = BRAIN_CONFIG.get("action_branches", [])
        if not isinstance(branch_sizes, (list, tuple)):
            branch_sizes = []
        self.action_branches = [int(size) for size in branch_sizes if int(size) > 0]
        self.branch_layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.transformer_embed, size) for size in self.action_branches]
        )
        self.transformer_heads = max(1, int(BRAIN_CONFIG["transformer_heads"]))
        configured_bot_heads = max(1, int(BRAIN_CONFIG.get("bot_heads", self.transformer_heads)))
        self.bot_heads = math.gcd(configured_bot_heads, self.transformer_embed) or 1
        self.head_dim = self.transformer_embed // self.bot_heads
        self.actions_per_head = max(1, math.ceil(self.action_count / self.bot_heads))
        self.register_buffer(
            "_meta_stub",
            torch.zeros(1, dtype=FLOAT_POLICY["dtype"]),
            persistent=False,
        )

    def forward(
        self, inputs: torch.Tensor, memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.dim() != 2:
            raise ValueError(f"Expected 2D inputs (batch, features); got shape {tuple(inputs.shape)}")
        use_batch_stats = self.training and inputs.size(0) > 1
        running_mean = self.input_bn.running_mean if self.input_bn.track_running_stats else None
        running_var = self.input_bn.running_var if self.input_bn.track_running_stats else None
        x = torch.nn.functional.batch_norm(
            inputs,
            running_mean,
            running_var,
            self.input_bn.weight,
            self.input_bn.bias,
            use_batch_stats,
            self.input_bn.momentum,
            self.input_bn.eps,
        )
        x = self.storsky_1(x)
        x = self.storsky_2(x)
        if memory_state is None or memory_state.numel() != self.memory_dim:
            memory_state = torch.zeros(self.memory_dim, device=x.device, dtype=x.dtype)
        memory_state = memory_state.to(device=x.device, dtype=x.dtype)
        batch_summary = x.mean(dim=0)
        memory_decay = float(min(max(self.memory_decay, 0.0), 0.999))
        new_memory = memory_decay * memory_state + (1.0 - memory_decay) * batch_summary
        memory_features = self.memory_project(self.memory_norm(new_memory))
        if self.memory_scale != 1.0:
            memory_features = memory_features * self.memory_scale
        x = x + memory_features.unsqueeze(0)
        transformer_state = self.transformer_project(x)
        transformer_state = transformer_state.view(x.size(0), self.transformer_tokens, self.transformer_embed)
        transformer_state = self.transformer_input_norm(transformer_state)
        transformer_state = self.transformer_encoder(transformer_state)
        transformer_state = self.transformer_output_norm(transformer_state)
        transformer_state = self.transformer_dense(transformer_state)
        transformer_state = self.transformer_dropout(transformer_state)
        if self.branch_layers:
            pooled = transformer_state.mean(dim=1)
            branch_logits = [layer(pooled) for layer in self.branch_layers]
            combined_logits = torch.cat(branch_logits, dim=-1)
            if combined_logits.size(-1) < self.action_count:
                pad = torch.zeros(
                    combined_logits.size(0),
                    self.action_count - combined_logits.size(-1),
                    device=combined_logits.device,
                    dtype=combined_logits.dtype,
                )
                combined_logits = torch.cat([combined_logits, pad], dim=-1)
            policy_logits = combined_logits[:, : self.action_count]
        else:
            transformer_state = transformer_state.reshape(
                x.size(0), self.transformer_tokens, self.bot_heads, self.head_dim
            )
            head_features = transformer_state.permute(0, 2, 1, 3)
            head_features = head_features.reshape(
                x.size(0), self.bot_heads, self.transformer_tokens * self.head_dim
            )
            if head_features.size(-1) < self.actions_per_head:
                pad = torch.zeros(
                    head_features.size(0),
                    head_features.size(1),
                    self.actions_per_head - head_features.size(-1),
                    device=head_features.device,
                    dtype=head_features.dtype,
                )
                head_features = torch.cat([head_features, pad], dim=-1)
            else:
                head_features = head_features[:, :, : self.actions_per_head]
            flat_features = head_features.reshape(head_features.size(0), -1)
            if flat_features.size(-1) < self.action_count:
                pad = torch.zeros(
                    flat_features.size(0),
                    self.action_count - flat_features.size(-1),
                    device=flat_features.device,
                    dtype=flat_features.dtype,
                )
                flat_features = torch.cat([flat_features, pad], dim=-1)
            policy_logits = flat_features[:, : self.action_count]
        probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        reward_prediction = policy_logits
        return probs, reward_prediction, new_memory

    def meta_state_norm(self) -> torch.Tensor:
        return self._meta_stub


class RemoteBrain:
    def __init__(self, input_size: int, action_count: int):
        self.input_size = int(input_size)
        self.action_count = int(action_count)
        self.model = BrainModel(self.input_size, self.action_count).to(FLOAT_POLICY["device"])
        self.memory_state = torch.zeros(
            self.model.memory_dim,
            device=FLOAT_POLICY["device"],
            dtype=FLOAT_POLICY["dtype"],
        )
        self.optimizer = LookaheadOptimizer(
            torch.optim.Adam(
                self.model.parameters(),
                lr=BRAIN_CONFIG["learning_rate"],
                betas=(BRAIN_CONFIG["lion_beta_1"], BRAIN_CONFIG["lion_beta_2"]),
                weight_decay=BRAIN_CONFIG["lion_weight_decay"],
            ),
            sync_period=BRAIN_CONFIG["lookahead_sync"],
            slow_step_size=BRAIN_CONFIG["lookahead_alpha"],
        )
        if hasattr(torch.cuda, "amp"):
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=FLOAT_POLICY["using_gpu"])
        else:  # pragma: no cover - fallback for older torch
            self.grad_scaler = None
        self.ema = ParameterEMA(self.model, decay=BRAIN_CONFIG["ema_decay"])
        self._lock = threading.RLock()
        self.policy_strikes = 0
        self.policy_last_reset = time.time()

    @property
    def device(self) -> torch.device:
        return FLOAT_POLICY["device"]

    def _autocast_context(self):
        if FLOAT_POLICY["using_gpu"] and hasattr(torch.cuda, "amp"):
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    def _prepare_observation(self, vector: Any, label: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if vector is None:
            raise ValueError(f"{label} payload missing")
        try:
            arr = np.asarray(vector, dtype=NP_FLOAT).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"{label} payload missing") from None
        invalid_mask = ~np.isfinite(arr)
        replaced = int(np.count_nonzero(invalid_mask))
        if replaced:
            arr = np.where(invalid_mask, 0.0, arr)
        clipped = 0
        if OBS_CLAMP > 0:
            clip_mask = np.abs(arr) > OBS_CLAMP
            clipped = int(np.count_nonzero(clip_mask))
            if clipped:
                arr = np.clip(arr, -OBS_CLAMP, OBS_CLAMP)
        adjusted = False
        if arr.size != self.input_size:
            adjusted = True
            if arr.size > self.input_size:
                arr = arr[: self.input_size]
            else:
                arr = np.pad(arr, (0, self.input_size - arr.size), constant_values=0.0)
        metadata = {
            "replaced": replaced,
            "clipped": clipped,
            "adjusted": adjusted,
            "sanitized": bool(replaced or clipped or adjusted),
        }
        return arr.astype(NP_FLOAT, copy=False), metadata

    def _weights_are_finite(self) -> bool:
        for param in self.model.parameters():
            if not torch.isfinite(param).all():
                return False
        return True

    def _sanitize_model_weights(self, reason: str) -> Dict[str, Any]:
        replaced_total = 0
        clipped_total = 0
        sanitized = False
        for param in self.model.parameters():
            tensor = param.data
            finite_mask = torch.isfinite(tensor)
            replaced = (~finite_mask).sum().item()
            if replaced:
                tensor[~finite_mask] = 0.0
            replaced_total += int(replaced)
            if WEIGHT_CLAMP > 0:
                clip_mask = tensor.abs() > WEIGHT_CLAMP
                if clip_mask.any():
                    tensor.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
                    clipped_total += int(clip_mask.sum().item())
            if replaced or clipped_total:
                sanitized = True
        if sanitized:
            LOGGER.warning(
                "Sanitized %s weights (replaced=%d, clipped=%d)",
                reason,
                replaced_total,
                clipped_total,
            )
            self.optimizer.sync_slow_parameters(list(self.model.parameters()))
            if hasattr(self, "ema"):
                self.ema.sync(self.model)
        return {
            "sanitized": sanitized,
            "replaced": replaced_total,
            "clipped": clipped_total,
        }

    def _filter_gradients(self) -> Tuple[List[torch.nn.Parameter], int, int, Optional[float]]:
        dropped = 0
        clipped = 0
        params = [p for p in self.model.parameters() if p.grad is not None]
        for param in list(params):
            grad = param.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                param.grad = None
                params.remove(param)
                dropped += 1
                LOGGER.warning("Dropped non-finite gradients for parameter %s", getattr(param, "name", "?"))
        if not params:
            return [], dropped, clipped, None
        if GRAD_CLIP_VALUE > 0:
            for param in params:
                grad = param.grad
                if grad is None:
                    continue
                if grad.abs().gt(GRAD_CLIP_VALUE).any():
                    grad.data.clamp_(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)
                    clipped += 1
        global_norm: Optional[float] = None
        try:
            if GRAD_CLIP_GLOBAL_NORM > 0:
                norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_GLOBAL_NORM)
                global_norm = float(norm)
                if global_norm > GRAD_CLIP_GLOBAL_NORM:
                    clipped += 1
            else:
                total = sum(torch.sum(param.grad.detach() ** 2) for param in params if param.grad is not None)
                global_norm = float(torch.sqrt(total).item()) if total > 0 else 0.0
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to compute gradient norm: %s", exc)
            global_norm = None
        if global_norm is not None:
            if not np.isfinite(global_norm):
                LOGGER.warning("Global gradient norm became non-finite; skipping step")
                for param in params:
                    param.grad = None
                return [], dropped, clipped, None
            if GRAD_SKIP_GLOBAL_NORM > 0 and global_norm > GRAD_SKIP_GLOBAL_NORM:
                LOGGER.warning(
                    "Global gradient norm %.3f exceeded skip threshold %.3f; skipping gradient application",
                    global_norm,
                    GRAD_SKIP_GLOBAL_NORM,
                )
                for param in params:
                    param.grad = None
                return [], dropped, clipped, global_norm
        return params, dropped, clipped, global_norm

    def _tensor_from_array(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array.astype(NP_FLOAT, copy=False)).to(self.device)
        return tensor

    def _get_meta_state_norm(self) -> float:
        meta_ref = getattr(self.model, "meta_state_norm", None)
        if callable(meta_ref):
            value = meta_ref()
        else:
            value = meta_ref
        if isinstance(value, torch.Tensor):
            try:
                return float(value.detach().abs().max().item())
            except Exception:  # pragma: no cover - defensive
                return 0.0
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(self.memory_state, torch.Tensor):
            try:
                return float(self.memory_state.detach().abs().max().item())
            except Exception:  # pragma: no cover - defensive
                return 0.0
        return 0.0

    def choose_actions_batch(self, observations: List[Any], epsilons: List[float]) -> List[Dict[str, Any]]:
        with self._lock:
            self.model.eval()
            results: List[Optional[Dict[str, Any]]] = [None] * len(observations)
            valid_entries = []

            for index, observation in enumerate(observations):
                epsilon = 0.1
                if index < len(epsilons):
                    try:
                        epsilon = float(epsilons[index])
                    except (TypeError, ValueError):
                        epsilon = 0.1
                try:
                    obs, obs_meta = self._prepare_observation(observation, "observation")
                except ValueError as exc:
                    results[index] = {"error": str(exc)}
                    continue
                valid_entries.append(
                    {
                        "index": index,
                        "epsilon": float(np.clip(epsilon, 0.0, 0.999)),
                        "tensor": obs,
                        "meta": obs_meta,
                    }
                )
            if valid_entries:
                batch = np.stack([entry["tensor"] for entry in valid_entries], axis=0).astype(NP_FLOAT, copy=False)
                meta_state_norm = 0.0
                reward_predictions_np = None
                with torch.no_grad():
                    ema_applied = False
                    try:
                        if hasattr(self, "ema") and self.ema is not None:
                            self.ema.apply_shadow(self.model)
                            ema_applied = True
                        tensor = torch.from_numpy(batch).to(self.device)
                        with self._autocast_context():
                            policy_probs, reward_predictions, memory_state = self.model(tensor, self.memory_state)
                        self.memory_state = memory_state.detach()
                        policy_probs = policy_probs.float()
                        reward_predictions = reward_predictions.float()
                        probs = policy_probs.cpu().numpy().astype(NP_FLOAT, copy=False)
                        reward_predictions_np = reward_predictions.cpu().numpy().astype(NP_FLOAT, copy=False)
                        meta_state_norm = self._get_meta_state_norm()
                    finally:
                        if ema_applied:
                            self.ema.restore(self.model)
                if reward_predictions_np is None:
                    reward_predictions_np = np.zeros((len(valid_entries), self.action_count), dtype=NP_FLOAT)
                for entry, row, reward_row in zip(valid_entries, probs, reward_predictions_np):
                    raw_probs = row.reshape(-1)
                    replaced = int(raw_probs.size - np.count_nonzero(np.isfinite(raw_probs)))
                    sanitized_policy = replaced > 0
                    safe_probs = np.nan_to_num(
                        raw_probs,
                        nan=1.0 / max(1, self.action_count),
                        posinf=1.0,
                        neginf=1.0,
                    )
                    safe_probs = np.clip(safe_probs, 1e-8, 1.0)
                    total = float(np.sum(safe_probs))
                    if not np.isfinite(total) or total <= 0:
                        sanitized_policy = True
                        safe_probs = np.full(self.action_count, 1.0 / max(1, self.action_count), dtype=NP_FLOAT)
                    else:
                        safe_probs = safe_probs / total
                    if not np.all(np.isfinite(safe_probs)):
                        sanitized_policy = True
                        safe_probs = np.full(self.action_count, 1.0 / max(1, self.action_count), dtype=NP_FLOAT)
                    now = time.time()
                    weights_ok = replaced == 0
                    strikes: Optional[int] = None
                    reward_row = reward_row.reshape(-1)
                    reward_replaced = int(reward_row.size - np.count_nonzero(np.isfinite(reward_row)))
                    reward_sanitized = reward_replaced > 0
                    safe_rewards = np.nan_to_num(reward_row, nan=0.0, posinf=0.0, neginf=0.0)
                    if safe_rewards.size != self.action_count:
                        reward_sanitized = True
                        if safe_rewards.size > self.action_count:
                            safe_rewards = safe_rewards[: self.action_count]
                        else:
                            safe_rewards = np.pad(
                                safe_rewards,
                                (0, self.action_count - safe_rewards.size),
                                constant_values=0.0,
                            )
                    if sanitized_policy:
                        LOGGER.warning("Sanitized action probabilities due to non-finite values")
                        if now - self.policy_last_reset > ACT_SANITIZATION_STRIKE_WINDOW:
                            self.policy_strikes = 0
                            self.policy_last_reset = now
                        self.policy_strikes += 1
                        strikes = self.policy_strikes
                        weight_meta = self._sanitize_model_weights("act-policy")
                        if weight_meta.get("sanitized"):
                            weights_ok = weights_ok and weight_meta.get("replaced", 0) == 0
                        weights_ok = weights_ok and self._weights_are_finite()
                        if strikes >= ACT_SANITIZATION_FORCE_THRESHOLD:
                            LOGGER.warning(
                                "Policy sanitization strike threshold reached (%d >= %d); flagging weights suspect.",
                                strikes,
                                ACT_SANITIZATION_FORCE_THRESHOLD,
                            )
                            weights_ok = False
                    else:
                        if now - self.policy_last_reset > ACT_SANITIZATION_STRIKE_WINDOW:
                            self.policy_strikes = 0
                            self.policy_last_reset = now
                        elif self.policy_strikes > 0:
                            self.policy_strikes = max(0, self.policy_strikes - 1)
                    exploration = float(np.random.random()) < entry["epsilon"]
                    if exploration:
                        action_index = int(np.random.randint(0, self.action_count))
                    else:
                        action_index = int(np.argmax(safe_probs))
                    policy_meta = {
                        "replaced": int(replaced),
                        "fallback": bool(sanitized_policy),
                        "sanitized": bool(replaced or sanitized_policy),
                    }
                    if strikes is not None:
                        policy_meta["strikes"] = int(strikes)
                        policy_meta["strike_threshold"] = int(ACT_SANITIZATION_FORCE_THRESHOLD)
                        policy_meta["window_ms"] = int(ACT_SANITIZATION_STRIKE_WINDOW * 1000)
                    action_prediction = 0.0
                    if 0 <= action_index < safe_rewards.size:
                        action_prediction = float(safe_rewards[action_index])
                    results[entry["index"]] = {
                        "action": action_index,
                        "weights_ok": bool(weights_ok),
                        "sanitized": {
                            "observation": entry["meta"],
                            "policy": policy_meta,
                            "exploration": bool(exploration),
                        },
                        "reward_prediction": {
                            "value": action_prediction,
                            "sanitized": bool(reward_sanitized),
                        },
                        "meta": {
                            "state_norm": float(meta_state_norm),
                        },
                    }
            final_results = [result or {"error": "Observation could not be processed"} for result in results]
        return final_results

    def choose_action(self, observation: Any, epsilon: float) -> Dict[str, Any]:
        batch = self.choose_actions_batch([observation], [epsilon])
        result = batch[0]
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    @staticmethod
    def _sanitize_scalar(value: Any, label: str) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            LOGGER.warning("Received invalid %s value: %s", label, value)
            return 0.0
        if not np.isfinite(numeric):
            LOGGER.warning("%s contained non-finite value: %s", label, value)
            return 0.0
        if numeric < 0.0:
            LOGGER.warning("%s was negative; clamping to zero: %s", label, value)
            numeric = 0.0
        return float(numeric)

    def _transform_reward(self, reward_value: Any, penalty_value: Any) -> Tuple[float, Dict[str, Any]]:
        reward_component = self._sanitize_scalar(reward_value, "reward")
        penalty_component = self._sanitize_scalar(penalty_value, "penalty")
        positive_component = max(0.0, reward_component)
        log_reward_component = np.log1p(positive_component)
        log_penalty_component = np.log1p(penalty_component)
        scaled_reward = log_reward_component - log_penalty_component
        return float(scaled_reward), {
            "raw": reward_component,
            "positive": positive_component,
            "penalty": penalty_component,
            "log_positive": float(log_reward_component),
            "log_penalty": float(log_penalty_component),
        }

    def train_batch(
        self,
        observations: List[Any],
        actions: List[Any],
        rewards: List[Any],
        penalties: List[Any],
        next_observations: List[Any],
    ) -> List[Dict[str, Any]]:
        with self._lock:
            self.model.train()
            results: List[Optional[Dict[str, Any]]] = [None] * len(observations)
            valid_entries = []

            def _sanitize_component(value, label):
                if value is None:
                    return 0.0
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    LOGGER.warning("Received invalid %s value: %s", label, value)
                    return 0.0
                if not np.isfinite(numeric):
                    LOGGER.warning("%s contained non-finite value: %s", label, value)
                    return 0.0
                if numeric < 0.0:
                    LOGGER.warning("%s was negative; clamping to zero: %s", label, value)
                    numeric = 0.0
                return float(numeric)

            for index, observation in enumerate(observations):
                try:
                    obs, obs_meta = self._prepare_observation(observation, "observation")
                except ValueError:
                    LOGGER.warning("Skipping train call due to missing observation payload")
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "sanitized": {
                            "observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                            "next_observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                            "weights": {"replaced": 0, "clipped": 0, "sanitized": False},
                        },
                        "reward_prediction": {
                            "predicted": 0.0,
                            "loss": None,
                            "advantage": 0.0,
                        },
                        "meta": {
                            "state_norm": self._get_meta_state_norm(),
                            "policy_loss": 0.0,
                        },
                    }
                    continue
                next_meta = {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False}
                next_obs = None
                if index < len(next_observations) and isinstance(next_observations[index], (list, tuple)):
                    next_obs, next_meta = self._prepare_observation(next_observations[index], "next_observation")
                action_value = actions[index] if index < len(actions) else None
                reward_value = rewards[index] if index < len(rewards) else None
                penalty_value = penalties[index] if index < len(penalties) else None
                if action_value is None and reward_value is None and penalty_value is None:
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "sanitized": {
                            "observation": obs_meta,
                            "next_observation": next_meta,
                            "weights": {"replaced": 0, "clipped": 0, "sanitized": False},
                            "reward": {
                                "raw": 0.0,
                                "positive": 0.0,
                                "penalty": 0.0,
                                "log_positive": 0.0,
                                "log_penalty": 0.0,
                            },
                        },
                        "reward_prediction": {
                            "predicted": 0.0,
                            "loss": None,
                            "advantage": 0.0,
                        },
                        "meta": {
                            "state_norm": self._get_meta_state_norm(),
                            "policy_loss": 0.0,
                        },
                    }
                    continue
                try:
                    action_index = int(action_value) if action_value is not None else None
                except (TypeError, ValueError):
                    action_index = None
                if action_index is None or action_index < 0 or action_index >= self.action_count:
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "sanitized": {
                            "observation": obs_meta,
                            "next_observation": next_meta,
                            "weights": {"replaced": 0, "clipped": 0, "sanitized": False},
                            "reward": {
                                "raw": 0.0,
                                "positive": 0.0,
                                "penalty": 0.0,
                                "log_positive": 0.0,
                                "log_penalty": 0.0,
                            },
                        },
                        "reward_prediction": {
                            "predicted": 0.0,
                            "loss": None,
                            "advantage": 0.0,
                        },
                        "meta": {
                            "state_norm": self._get_meta_state_norm(),
                            "policy_loss": 0.0,
                        },
                    }
                    continue
                scaled_reward, reward_meta = self._transform_reward(reward_value, penalty_value)
                valid_entries.append(
                    {
                        "index": index,
                        "obs": obs,
                        "action": action_index,
                        "reward": scaled_reward,
                        "obs_meta": obs_meta,
                        "next_meta": next_meta,
                        "reward_meta": reward_meta,
                    }
                )
            if valid_entries:
                obs_matrix = np.stack([entry["obs"] for entry in valid_entries], axis=0).astype(NP_FLOAT, copy=False)
                action_tensor = torch.tensor([entry["action"] for entry in valid_entries], dtype=torch.long, device=self.device)
                reward_tensor = torch.tensor([entry["reward"] for entry in valid_entries], dtype=FLOAT_POLICY["dtype"], device=self.device)
                obs_tensor = torch.from_numpy(obs_matrix).to(self.device)
                self.optimizer.zero_grad()
                with self._autocast_context():
                    policy_probs, reward_predictions, memory_state = self.model(obs_tensor, self.memory_state)
                self.memory_state = memory_state.detach()
                policy_probs = policy_probs.float()
                reward_predictions = reward_predictions.float()
                log_probs = torch.log(policy_probs + 1e-8)
                selected_log_probs = log_probs.gather(1, action_tensor.view(-1, 1)).squeeze(1)
                predicted_rewards = reward_predictions.gather(1, action_tensor.view(-1, 1)).squeeze(1)
                advantages = reward_tensor - predicted_rewards.detach()
                advantages = advantages - advantages.mean()
                policy_loss = -(selected_log_probs * advantages).mean()
                prediction_loss = torch.nn.functional.smooth_l1_loss(predicted_rewards, reward_tensor)
                prediction_weight = float(BRAIN_CONFIG.get("reward_prediction_weight", 0.0))
                total_loss = policy_loss + prediction_weight * prediction_loss
                params: List[torch.nn.Parameter] = []
                dropped = 0
                clipped_grads = 0
                gradient_norm: Optional[float] = None
                weights_ok = True
                weight_meta = {"sanitized": False, "replaced": 0, "clipped": 0}
                policy_loss_value = float(policy_loss.detach().cpu().item())
                prediction_loss_value = float(prediction_loss.detach().cpu().item())
                meta_state_norm = self._get_meta_state_norm()
                if not torch.isfinite(total_loss):
                    LOGGER.warning("Loss became non-finite; skipping update")
                    if getattr(self, "grad_scaler", None) and self.grad_scaler.is_enabled():
                        self.grad_scaler.update()
                else:
                    scaler = self.grad_scaler if getattr(self, "grad_scaler", None) and self.grad_scaler.is_enabled() else None
                    if scaler is not None:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(self.optimizer.optimizer)
                    else:
                        total_loss.backward()
                    params, dropped, clipped_grads, gradient_norm = self._filter_gradients()
                    if params:
                        self.optimizer.step(scaler)
                        weights_ok = self._weights_are_finite()
                        if not weights_ok:
                            LOGGER.error("Model weights contain non-finite values after training step")
                        weight_meta = self._sanitize_model_weights("post-train-batch")
                        if weight_meta.get("sanitized"):
                            weights_ok = weights_ok and weight_meta.get("replaced", 0) == 0 and self._weights_are_finite()
                        if hasattr(self, "ema") and self.ema is not None:
                            if weight_meta.get("sanitized"):
                                self.ema.sync(self.model)
                            else:
                                self.ema.update(self.model)
                    else:
                        if scaler is not None:
                            scaler.update()
                        dropped = max(dropped, 1)
                        gradient_norm = None
                        self.optimizer.zero_grad()
                        weights_ok = self._weights_are_finite()
                predicted_values_np = predicted_rewards.detach().cpu().numpy().astype(NP_FLOAT, copy=False)
                advantages_np = advantages.detach().cpu().numpy().astype(NP_FLOAT, copy=False)
                for offset, entry in enumerate(valid_entries):
                    predicted_value = float(predicted_values_np[offset]) if offset < len(predicted_values_np) else 0.0
                    advantage_value = float(advantages_np[offset]) if offset < len(advantages_np) else 0.0
                    results[entry["index"]] = {
                        "trained": bool(params),
                        "weights_ok": bool(weights_ok),
                        "dropped_gradients": int(dropped),
                        "clipped_gradients": int(clipped_grads),
                        "gradient_norm": float(gradient_norm) if gradient_norm is not None else None,
                        "sanitized": {
                            "observation": entry["obs_meta"],
                            "next_observation": entry["next_meta"],
                            "weights": {
                                "replaced": int(weight_meta.get("replaced", 0)),
                                "clipped": int(weight_meta.get("clipped", 0)),
                                "sanitized": bool(weight_meta.get("sanitized", False)),
                            },
                            "reward": entry["reward_meta"],
                        },
                        "reward_prediction": {
                            "predicted": predicted_value,
                            "loss": prediction_loss_value,
                            "advantage": advantage_value,
                        },
                        "meta": {
                            "state_norm": float(meta_state_norm),
                            "policy_loss": policy_loss_value,
                        },
                    }
            idle_meta_norm = self._get_meta_state_norm()
            final_results = [
                value
                if value is not None
                else {
                    "trained": False,
                    "weights_ok": True,
                    "dropped_gradients": 0,
                    "clipped_gradients": 0,
                    "gradient_norm": None,
                    "sanitized": {
                        "observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                        "next_observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                        "weights": {"replaced": 0, "clipped": 0, "sanitized": False},
                        "reward": {
                            "raw": 0.0,
                            "positive": 0.0,
                            "penalty": 0.0,
                            "log_positive": 0.0,
                            "log_penalty": 0.0,
                        },
                    },
                    "reward_prediction": {
                        "predicted": 0.0,
                        "loss": None,
                        "advantage": 0.0,
                    },
                    "meta": {
                        "state_norm": float(idle_meta_norm),
                        "policy_loss": 0.0,
                    },
                }
                for value in results
            ]
        return final_results

    def train(self, observation: Any, action_index: Any, reward: Any, penalty: Any, next_observation: Any) -> Dict[str, Any]:
        batch = self.train_batch([observation], [action_index], [reward], [penalty], [next_observation])
        return batch[0]

    def copy_from(self, other: "RemoteBrain") -> None:
        if other is self:
            return
        first, second = (self, other) if id(self) <= id(other) else (other, self)
        with first._lock:
            with second._lock:
                self.model.load_state_dict(other.model.state_dict())
                self.optimizer.sync_slow_parameters(list(self.model.parameters()))
                if hasattr(self, "ema") and self.ema is not None:
                    self.ema.sync(self.model)

    def average_from(self, sources: List["RemoteBrain"]) -> None:
        if not sources:
            return
        with self._lock:
            state_dicts = []
            for source in sources:
                if source is None:
                    continue
                with source._lock:
                    state_dicts.append({key: value.detach().clone() for key, value in source.model.state_dict().items()})
            if not state_dicts:
                return
            averaged = {}
            for key in state_dicts[0]:
                stacked = torch.stack([state[key] for state in state_dicts], dim=0)
                if stacked.dtype.is_floating_point or stacked.dtype.is_complex:
                    averaged_value = stacked.mean(dim=0)
                else:
                    averaged_value = torch.round(stacked.to(torch.float32).mean(dim=0)).to(stacked.dtype)
                averaged[key] = averaged_value
            self.model.load_state_dict(averaged)
            self.optimizer.sync_slow_parameters(list(self.model.parameters()))
            if hasattr(self, "ema") and self.ema is not None:
                self.ema.sync(self.model)

    def mutate(self, stddev: float) -> None:
        with self._lock:
            for param in self.model.parameters():
                noise = torch.randn_like(param) * stddev
                param.data.add_(noise)
            self._sanitize_model_weights("mutate")
            self.optimizer.sync_slow_parameters(list(self.model.parameters()))
            if hasattr(self, "ema") and self.ema is not None:
                self.ema.sync(self.model)

    def save(self, directory: str) -> None:
        with self._lock:
            os.makedirs(directory, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
            metadata = {
                "input_size": self.input_size,
                "action_count": self.action_count,
                "created_at": time.time(),
                "dtype": str(FLOAT_POLICY["dtype"]),
            }
            with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

    def load_weights(self, directory: str) -> Dict[str, Any]:
        path = os.path.join(directory, "model.pt")
        if not os.path.exists(path):
            return {"status": "fresh", "path": path}
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.sync_slow_parameters(list(self.model.parameters()))
        meta = self._sanitize_model_weights("load")
        if hasattr(self, "ema") and self.ema is not None:
            self.ema.sync(self.model)
        status = "exact"
        if meta.get("sanitized"):
            status = "sanitized"
        return {"status": status, "path": path, "weights": meta}


def require_brain(brain_id: str) -> RemoteBrain:
    brain = BRAINS.get(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Unknown brain")
    return brain


def _execute(endpoint: str, func, *args, **kwargs):
    try:
        return WORKERS.run(endpoint, func, *args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Endpoint '%s' task failed", endpoint)
        raise RuntimeError(f"Failed to process {endpoint} request") from exc
    finally:
        MEMORY_MANAGER.maybe_collect()


async def _async_execute(endpoint: str, func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    bound = functools.partial(_execute, endpoint, func, *args, **kwargs)
    return await loop.run_in_executor(None, bound)


class StreamProtocolError(Exception):
    """Raised when a streaming request is malformed."""


def _encode_stream_response(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


async def _stream_handle_ping(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"pong": True, "timestamp": time.time()}


async def _stream_handle_act(message: Dict[str, Any]) -> Dict[str, Any]:
    brain_id = message.get("brain_id")
    if not brain_id:
        raise StreamProtocolError("brain_id is required")
    brain = require_brain(brain_id)
    observation = message.get("observation")
    if not isinstance(observation, (list, tuple)):
        raise StreamProtocolError("observation must be a list")
    epsilon = message.get("epsilon", 0.1)
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError):
        epsilon_value = 0.1
    result = await _async_execute("act", brain.choose_action, observation, epsilon_value)
    response_payload = {
        "action": int(result.get("action", 0)),
        "weights_ok": bool(result.get("weights_ok", True)),
        "sanitized": result.get("sanitized") or {},
    }
    if "reward_prediction" in result:
        response_payload["reward_prediction"] = result["reward_prediction"]
    if "meta" in result:
        response_payload["meta"] = result["meta"]
    request_payload = {key: value for key, value in message.items() if key not in {"id", "type"}}
    log_request(message.get("bot_id"), "act", {**request_payload, **response_payload})
    return response_payload


async def _stream_handle_train(message: Dict[str, Any]) -> Dict[str, Any]:
    brain_id = message.get("brain_id")
    if not brain_id:
        raise StreamProtocolError("brain_id is required")
    brain = require_brain(brain_id)
    observation = message.get("observation")
    if not isinstance(observation, (list, tuple)):
        raise StreamProtocolError("observation must be a list")
    next_observation = message.get("next_observation")
    if not isinstance(next_observation, (list, tuple)):
        next_observation = None
    result = await _async_execute(
        "train",
        brain.train,
        observation,
        message.get("action"),
        message.get("reward"),
        message.get("penalty"),
        next_observation,
    )
    response_payload = {
        "trained": bool(result.get("trained")),
        "weights_ok": bool(result.get("weights_ok", True)),
        "dropped_gradients": int(result.get("dropped_gradients", 0)),
        "sanitized": result.get("sanitized") or {},
    }
    if "clipped_gradients" in result:
        response_payload["clipped_gradients"] = int(result.get("clipped_gradients", 0))
    if "gradient_norm" in result:
        response_payload["gradient_norm"] = result.get("gradient_norm")
    if "reward_prediction" in result:
        response_payload["reward_prediction"] = result["reward_prediction"]
    if "meta" in result:
        response_payload["meta"] = result["meta"]
    if "learning_rate" in result:
        response_payload["learning_rate"] = result["learning_rate"]
    request_payload = {key: value for key, value in message.items() if key not in {"id", "type"}}
    log_request(message.get("bot_id"), "train", {**request_payload, **response_payload})
    return response_payload


STREAM_HANDLERS = {
    "ping": _stream_handle_ping,
    "act": _stream_handle_act,
    "train": _stream_handle_train,
}


async def _dispatch_stream_message(message: Dict[str, Any]) -> bytes:
    message_id = message.get("id")
    message_type = message.get("type")
    handler = STREAM_HANDLERS.get(message_type)
    if handler is None:
        error_payload = {
            "id": message_id,
            "type": message_type,
            "ok": False,
            "error": {"message": f"Unknown message type: {message_type}"},
        }
        return _encode_stream_response(error_payload)
    try:
        result = await handler(message)
        payload = {"id": message_id, "type": message_type, "ok": True, "result": result}
    except StreamProtocolError as exc:
        payload = {
            "id": message_id,
            "type": message_type,
            "ok": False,
            "error": {"message": str(exc)},
        }
    except HTTPException as exc:
        payload = {
            "id": message_id,
            "type": message_type,
            "ok": False,
            "error": {"message": exc.detail, "status": exc.status_code},
        }
    except RuntimeError as exc:
        payload = {
            "id": message_id,
            "type": message_type,
            "ok": False,
            "error": {"message": str(exc)},
        }
    except Exception:  # pragma: no cover - safety net
        LOGGER.exception("Unhandled exception while processing stream message")
        payload = {
            "id": message_id,
            "type": message_type,
            "ok": False,
            "error": {"message": "Internal server error"},
        }
    return _encode_stream_response(payload)


async def _iter_stream_messages(request: Request):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    try:
        async for chunk in request.stream():
            buffer += decoder.decode(chunk, final=False)
            while True:
                newline_index = buffer.find("\n")
                if newline_index == -1:
                    break
                line = buffer[:newline_index].strip()
                buffer = buffer[newline_index + 1 :]
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    error_payload = {
                        "id": None,
                        "type": None,
                        "ok": False,
                        "error": {"message": f"Invalid JSON payload: {exc}"},
                    }
                    yield _encode_stream_response(error_payload)
                    continue
                yield await _dispatch_stream_message(message)
        tail = decoder.decode(b"", final=True)
        buffer += tail
        line = buffer.strip()
        if line:
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                error_payload = {
                    "id": None,
                    "type": None,
                    "ok": False,
                    "error": {"message": f"Invalid JSON payload: {exc}"},
                }
                yield _encode_stream_response(error_payload)
            else:
                yield await _dispatch_stream_message(message)
    finally:
        MEMORY_MANAGER.maybe_collect(force=True)


def log_request(bot_id: Any, endpoint: str, payload: Dict[str, Any]) -> None:
    with STATE_LOCK:
        STATUS["total_requests"] += 1
        entry = {
            "timestamp": time.time(),
            "bot_id": bot_id,
            "endpoint": endpoint,
            "reward": payload.get("reward"),
            "penalty": payload.get("penalty"),
            "action": payload.get("action"),
            "epsilon": payload.get("epsilon"),
            "gradient_norm": payload.get("gradient_norm"),
            "reward_prediction": payload.get("reward_prediction"),
        }
        STATUS["requests"].append(entry)
        STATUS["requests"] = STATUS["requests"][-50:]
        STATUS["workers"] = WORKERS.snapshot()
        if bot_id:
            bot_state = STATUS["bots"].setdefault(bot_id, {})
            bot_state.update(
                {
                    "last_seen": entry["timestamp"],
                    "last_action": payload.get("action"),
                    "last_reward": payload.get("reward"),
                    "last_penalty": payload.get("penalty"),
                    "epsilon": payload.get("epsilon"),
                    "gradient_norm": payload.get("gradient_norm"),
                }
            )
            sanitized = payload.get("sanitized")
            if isinstance(sanitized, dict):
                policy_meta = sanitized.get("policy")
                if isinstance(policy_meta, dict) and "strikes" in policy_meta:
                    bot_state["policy_strikes"] = int(policy_meta["strikes"])
                    bot_state["policy_strike_threshold"] = ACT_SANITIZATION_FORCE_THRESHOLD
                    bot_state["policy_last_sanitized"] = entry["timestamp"]


@app.post("/api/brains/stream")
async def brain_stream_endpoint(request: Request):
    return StreamingResponse(_iter_stream_messages(request), media_type="application/jsonl")


@app.post("/api/brains")
def create_brain_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    input_size = int(payload.get("input_size", 0))
    action_count = int(payload.get("action_count", 0))
    if input_size <= 0 or action_count <= 0:
        raise HTTPException(status_code=400, detail="Invalid brain dimensions")
    brain_id = str(uuid.uuid4())
    BRAINS[brain_id] = RemoteBrain(input_size, action_count)
    return {"brain_id": brain_id}


@app.post("/api/brains/{brain_id}/act")
def choose_action_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    observation = payload.get("observation")
    if not isinstance(observation, (list, tuple)):
        raise HTTPException(status_code=400, detail="Observation must be a list")
    epsilon = float(payload.get("epsilon", 0.1))
    try:
        result = _execute("act", brain.choose_action, observation, epsilon)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response_payload = {
        "action": int(result.get("action", 0)),
        "weights_ok": bool(result.get("weights_ok", True)),
        "sanitized": result.get("sanitized") or {},
    }
    if "reward_prediction" in result:
        response_payload["reward_prediction"] = result["reward_prediction"]
    if "meta" in result:
        response_payload["meta"] = result["meta"]
    log_request(payload.get("bot_id"), "act", {**payload, **response_payload})
    return response_payload


@app.post("/api/brains/{brain_id}/train")
def train_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    observation = payload.get("observation")
    if not isinstance(observation, (list, tuple)):
        raise HTTPException(status_code=400, detail="Observation must be a list")
    next_observation = payload.get("next_observation")
    try:
        result = _execute(
            "train",
            brain.train,
            observation,
            payload.get("action"),
            payload.get("reward"),
            payload.get("penalty"),
            next_observation if isinstance(next_observation, (list, tuple)) else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response_payload = {
        "trained": bool(result.get("trained")),
        "weights_ok": bool(result.get("weights_ok", True)),
        "dropped_gradients": int(result.get("dropped_gradients", 0)),
        "sanitized": result.get("sanitized") or {},
    }
    if "reward_prediction" in result:
        response_payload["reward_prediction"] = result["reward_prediction"]
    if "meta" in result:
        response_payload["meta"] = result["meta"]
    log_request(payload.get("bot_id"), "train", {**payload, **response_payload})
    return response_payload


@app.post("/api/brains/batch_act")
def batch_act_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="Requests must be a list")
    results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            results[index] = {"error": "Invalid request"}
            continue
        brain_id = item.get("brain_id")
        if not brain_id:
            results[index] = {"error": "Missing brain_id"}
            continue
        grouped.setdefault(brain_id, []).append((index, item))
    for brain_id, entries in grouped.items():
        brain = BRAINS.get(brain_id)
        if brain is None:
            for index, _ in entries:
                results[index] = {"error": "Unknown brain"}
            continue
        observations: List[List[Any]] = []
        epsilons: List[float] = []
        valid_indices: List[int] = []
        for index, item in entries:
            observation = item.get("observation")
            if not isinstance(observation, (list, tuple)):
                results[index] = {"error": "Observation must be a list"}
                continue
            observations.append(observation)
            epsilons.append(item.get("epsilon", 0.1))
            valid_indices.append(index)
        if not observations:
            continue
        try:
            brain_results = _execute("act", brain.choose_actions_batch, observations, epsilons)
        except RuntimeError as exc:
            error_payload = {"error": str(exc)}
            for index in valid_indices:
                results[index] = error_payload
            continue
        for offset, index in enumerate(valid_indices):
            entry_result = brain_results[offset] if offset < len(brain_results) else None
            if not isinstance(entry_result, dict) or entry_result.get("error"):
                results[index] = {
                    "error": entry_result.get("error") if isinstance(entry_result, dict) else "Failed to compute action",
                }
                continue
            response_payload = {
                "action": int(entry_result.get("action", 0)),
                "weights_ok": bool(entry_result.get("weights_ok", True)),
                "sanitized": entry_result.get("sanitized") or {},
            }
            if "reward_prediction" in entry_result:
                response_payload["reward_prediction"] = entry_result["reward_prediction"]
            if "meta" in entry_result:
                response_payload["meta"] = entry_result["meta"]
            results[index] = response_payload
        for index, item in entries:
            response_payload = results[index] or {}
            log_request(item.get("bot_id"), "act", {**item, **response_payload})
    for index, value in enumerate(results):
        if value is None:
            results[index] = {"error": "Request was not processed"}
    return {"results": results}


@app.post("/api/brains/batch_train")
def batch_train_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="Requests must be a list")
    results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            results[index] = {"error": "Invalid request"}
            continue
        brain_id = item.get("brain_id")
        if not brain_id:
            results[index] = {"error": "Missing brain_id"}
            continue
        grouped.setdefault(brain_id, []).append((index, item))
    for brain_id, entries in grouped.items():
        brain = BRAINS.get(brain_id)
        if brain is None:
            for index, _ in entries:
                results[index] = {"error": "Unknown brain"}
            continue
        observations: List[List[Any]] = []
        actions: List[Any] = []
        rewards: List[Any] = []
        penalties: List[Any] = []
        next_observations: List[Any] = []
        valid_indices: List[int] = []
        for index, item in entries:
            observation = item.get("observation")
            if not isinstance(observation, (list, tuple)):
                results[index] = {"error": "Observation must be a list"}
                continue
            observations.append(observation)
            actions.append(item.get("action"))
            rewards.append(item.get("reward"))
            penalties.append(item.get("penalty"))
            next_observations.append(item.get("next_observation"))
            valid_indices.append(index)
        if not observations:
            continue
        try:
            brain_results = _execute(
                "train",
                brain.train_batch,
                observations,
                actions,
                rewards,
                penalties,
                next_observations,
            )
        except RuntimeError as exc:
            error_payload = {"error": str(exc)}
            for index in valid_indices:
                results[index] = error_payload
            continue
        for offset, index in enumerate(valid_indices):
            entry_result = brain_results[offset] if offset < len(brain_results) else None
            if not isinstance(entry_result, dict):
                results[index] = {"error": "Failed to process training request"}
                continue
            response_payload = {
                "trained": bool(entry_result.get("trained")),
                "weights_ok": bool(entry_result.get("weights_ok", True)),
                "dropped_gradients": int(entry_result.get("dropped_gradients", 0)),
                "sanitized": entry_result.get("sanitized") or {},
            }
            if "reward_prediction" in entry_result:
                response_payload["reward_prediction"] = entry_result["reward_prediction"]
            if "meta" in entry_result:
                response_payload["meta"] = entry_result["meta"]
            results[index] = response_payload
        for index, item in entries:
            response_payload = results[index] or {}
            log_request(item.get("bot_id"), "train", {**item, **response_payload})
    for index, value in enumerate(results):
        if value is None:
            results[index] = {"error": "Request was not processed"}
    return {"results": results}


@app.post("/api/brains/{brain_id}/copy")
def copy_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    source_id = payload.get("source_id")
    if not source_id:
        raise HTTPException(status_code=400, detail="source_id is required")
    source = require_brain(source_id)
    try:
        _execute("copy", brain.copy_from, source)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/brains/{brain_id}/average")
def average_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    source_ids = payload.get("source_ids") or []
    sources = [require_brain(source_id) for source_id in source_ids]
    try:
        _execute("average", brain.average_from, sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/brains/{brain_id}/mutate")
def mutate_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    stddev = float(payload.get("stddev", 0.02))
    if stddev <= 0:
        raise HTTPException(status_code=400, detail="Stddev must be positive")
    try:
        _execute("mutate", brain.mutate, stddev)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/brains/{brain_id}/save")
def save_endpoint(brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    brain = require_brain(brain_id)
    payload = payload or {}
    label = payload.get("path") or brain_id
    target_dir = _resolve_storage_path("brains", label, create=True)
    try:
        _execute("save", brain.save, target_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "path": label}


@app.post("/api/brains/load")
def load_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    path_label = payload.get("path")
    input_size = int(payload.get("input_size", 0))
    action_count = int(payload.get("action_count", 0))
    if not path_label or input_size <= 0 or action_count <= 0:
        raise HTTPException(status_code=400, detail="Invalid load request")
    path = _resolve_storage_path("brains", path_label, create=False)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    brain_id = str(uuid.uuid4())
    brain = RemoteBrain(input_size, action_count)
    try:
        load_result = _execute("load", brain.load_weights, path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    BRAINS[brain_id] = brain
    response = {"brain_id": brain_id}
    if isinstance(load_result, dict):
        response.update({key: value for key, value in load_result.items() if key not in {"path"}})
        if load_result.get("status"):
            response["source_path"] = load_result.get("path")
    return response


@app.post("/api/state/save")
def save_state_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    label = payload.get("path")
    if not label:
        raise HTTPException(status_code=400, detail="Path is required")
    state = payload.get("state") or {}
    path = _resolve_storage_path("state", label, create=True)
    state_path = os.path.join(path, "brain_state.json")
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return {"status": "ok", "path": label}


@app.post("/api/state/load")
def load_state_endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
    payload = payload or {}
    label = payload.get("path")
    if not label:
        raise HTTPException(status_code=400, detail="Path is required")
    path = _resolve_storage_path("state", label, create=False)
    state_path = os.path.join(path, "brain_state.json")
    if not os.path.exists(state_path):
        return {"state": None}
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    return {"state": state}


@app.get("/status")
def status_endpoint():
    with STATE_LOCK:
        STATUS["workers"] = WORKERS.snapshot()
        payload = {
            "started_at": STATUS["started_at"],
            "uptime": time.time() - STATUS["started_at"],
            "total_requests": STATUS["total_requests"],
            "brain_count": len(BRAINS),
            "accelerators": STATUS.get("accelerators"),
            "float_policy": STATUS.get("float_policy"),
            "bots": list(STATUS.get("bots", {}).items()),
            "recent_requests": STATUS.get("requests", []),
            "workers": STATUS.get("workers", {}),
        }
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("TF_SERVER_PORT", 5000))
    uvicorn.run("tf_server:app", host="0.0.0.0", port=port, reload=False)
