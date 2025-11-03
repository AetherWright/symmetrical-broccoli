import atexit
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
from fastapi import Body, FastAPI, HTTPException
import uvicorn

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
    "hidden_units": 192,
    "mid_units": 144,
    "shared_units": 96,
    "dropout_rate": 0.3,
    "hebbian_units": [160, 112],
    "hebbian_learning_rate": 0.01,
    "hebbian_decay_multiplier": 1.5,
    "hebbian_clip": 0.75,
    "transformer_tokens": 6,
    "transformer_embed": 32,
    "transformer_heads": 4,
    "transformer_layers": 2,
    "transformer_ff": 256,
    "transformer_rotary_base": 10000.0,
    "learning_rate": 2e-3,
    "lion_beta_1": 0.9,
    "lion_beta_2": 0.99,
    "lion_weight_decay": 0.0,
    "lookahead_sync": 6,
    "lookahead_alpha": 0.5,
    "ema_decay": 0.995,
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

    def step(self) -> None:
        self._maybe_refresh()
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


class HebbianLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=natural_log_relu,
        hebbian_learning_rate: float = 0.01,
        decay_multiplier: float = 1.5,
        clip: float = 0.75,
    ):
        super().__init__()
        self.base = torch.nn.Linear(in_features, out_features)
        self.activation = activation
        self.hebbian_learning_rate = float(abs(hebbian_learning_rate))
        self.decay_multiplier = float(abs(decay_multiplier))
        self.clip = float(abs(clip))
        self.register_buffer("hebbian_weight", torch.zeros(out_features, in_features))
        self.register_buffer("hebbian_bias", torch.zeros(out_features))
        self._last_input: Optional[torch.Tensor] = None
        self._last_output: Optional[torch.Tensor] = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.base.weight + self.hebbian_weight
        bias = self.base.bias + self.hebbian_bias
        outputs = torch.nn.functional.linear(inputs, weight, bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.training:
            self._last_input = inputs.detach()
            self._last_output = outputs.detach()
        else:
            self._last_input = None
            self._last_output = None
        return outputs

    def _apply_decay(self) -> float:
        if self.hebbian_learning_rate <= 0:
            return 0.0
        decay_rate = self.hebbian_learning_rate * self.decay_multiplier
        decay_rate = min(0.95, max(0.0, decay_rate))
        if decay_rate == 0:
            return 0.0
        keep_ratio = 1.0 - decay_rate
        self.hebbian_weight.mul_(keep_ratio)
        self.hebbian_bias.mul_(keep_ratio)
        return float(decay_rate)

    def update_hebbian(self, reinforcement: float) -> Dict[str, Any]:
        decay_rate = self._apply_decay()
        reinforcement = float(np.clip(reinforcement, -1.0, 1.0))
        applied = False
        if reinforcement != 0.0 and self._last_input is not None and self._last_output is not None:
            lr = self.hebbian_learning_rate * reinforcement
            pre = torch.nn.functional.normalize(self._last_input, dim=-1)
            post = torch.nn.functional.normalize(self._last_output, dim=-1)
            kernel_update = torch.einsum("bi,bj->ij", post, pre)
            kernel_update /= max(1.0, float(pre.shape[0]))
            bias_update = post.mean(dim=0)
            self.hebbian_weight.add_(kernel_update * lr)
            self.hebbian_bias.add_(bias_update * lr)
            applied = True
        if self.clip > 0:
            self.hebbian_weight.clamp_(-self.clip, self.clip)
            self.hebbian_bias.clamp_(-self.clip, self.clip)
        self._last_input = None
        self._last_output = None
        return {"applied": applied, "decay_rate": decay_rate}


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
        self.hidden_dense_1 = torch.nn.Linear(input_size, BRAIN_CONFIG["hidden_units"])
        self.hidden_dropout_1 = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.hebbian_dense_1 = HebbianLinear(
            BRAIN_CONFIG["hidden_units"],
            BRAIN_CONFIG["hebbian_units"][0],
            hebbian_learning_rate=BRAIN_CONFIG["hebbian_learning_rate"],
            decay_multiplier=BRAIN_CONFIG["hebbian_decay_multiplier"],
            clip=BRAIN_CONFIG["hebbian_clip"],
        )
        self.hebbian_dropout_1 = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.hidden_dense_2 = torch.nn.Linear(BRAIN_CONFIG["hebbian_units"][0], BRAIN_CONFIG["mid_units"])
        self.hidden_dropout_2 = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.hebbian_dense_2 = HebbianLinear(
            BRAIN_CONFIG["mid_units"],
            BRAIN_CONFIG["hebbian_units"][1],
            hebbian_learning_rate=BRAIN_CONFIG["hebbian_learning_rate"],
            decay_multiplier=BRAIN_CONFIG["hebbian_decay_multiplier"],
            clip=BRAIN_CONFIG["hebbian_clip"],
        )
        self.shared_dense = torch.nn.Linear(BRAIN_CONFIG["hebbian_units"][1], BRAIN_CONFIG["shared_units"])
        self.shared_dropout = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.transformer_tokens = int(BRAIN_CONFIG["transformer_tokens"])
        self.transformer_embed = int(BRAIN_CONFIG["transformer_embed"])
        total_transformer_dim = self.transformer_tokens * self.transformer_embed
        if total_transformer_dim <= 0:
            raise ValueError("Transformer configuration must produce a positive feature size")
        if self.transformer_embed % max(1, int(BRAIN_CONFIG["transformer_heads"])) != 0:
            raise ValueError("Transformer embed dimension must be divisible by the number of heads")
        self.transformer_project = torch.nn.Linear(BRAIN_CONFIG["shared_units"], total_transformer_dim)
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
        self.transformer_merge = torch.nn.Linear(total_transformer_dim, BRAIN_CONFIG["shared_units"])
        self.context_gate = torch.nn.Linear(BRAIN_CONFIG["shared_units"] * 2, BRAIN_CONFIG["shared_units"])
        self.policy_dense = torch.nn.Linear(BRAIN_CONFIG["shared_units"], BRAIN_CONFIG["shared_units"])
        self.policy_dropout = torch.nn.Dropout(BRAIN_CONFIG["dropout_rate"])
        self.action_head = torch.nn.Linear(BRAIN_CONFIG["shared_units"], action_count)
        self.action_count = action_count

    @staticmethod
    def _activate(tensor: torch.Tensor) -> torch.Tensor:
        return natural_log_relu(tensor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        x = self._activate(self.hidden_dense_1(x))
        x = self.hidden_dropout_1(x)
        x = self.hebbian_dense_1(x)
        x = self.hebbian_dropout_1(x)
        x = self._activate(self.hidden_dense_2(x))
        x = self.hidden_dropout_2(x)
        x = self.hebbian_dense_2(x)
        shared = self._activate(self.shared_dense(x))
        shared = self.shared_dropout(shared)
        transformer_state = self.transformer_project(shared)
        transformer_state = transformer_state.view(
            shared.size(0), self.transformer_tokens, self.transformer_embed
        )
        transformer_state = self.transformer_input_norm(transformer_state)
        transformer_state = self.transformer_encoder(transformer_state)
        transformer_state = self.transformer_output_norm(transformer_state)
        transformer_state = self.transformer_dropout(transformer_state)
        transformer_state = transformer_state.reshape(shared.size(0), -1)
        transformer_features = self.transformer_merge(transformer_state)
        gate_input = torch.cat([shared, transformer_features], dim=-1)
        context_weights = torch.sigmoid(self.context_gate(gate_input))
        shared = shared + context_weights * transformer_features
        policy_features = self._activate(self.policy_dense(shared))
        policy_features = self.policy_dropout(policy_features)
        logits = self.action_head(policy_features)
        return torch.nn.functional.softmax(logits, dim=-1)

    def hebbian_layers(self) -> List[HebbianLinear]:
        return [self.hebbian_dense_1, self.hebbian_dense_2]


class RemoteBrain:
    def __init__(self, input_size: int, action_count: int):
        self.input_size = int(input_size)
        self.action_count = int(action_count)
        self.model = BrainModel(self.input_size, self.action_count).to(FLOAT_POLICY["device"])
        self.optimizer = LookaheadOptimizer(
            torch.optim.AdamW(
                self.model.parameters(),
                lr=BRAIN_CONFIG["learning_rate"],
                betas=(BRAIN_CONFIG["lion_beta_1"], BRAIN_CONFIG["lion_beta_2"]),
                weight_decay=BRAIN_CONFIG["lion_weight_decay"],
            ),
            sync_period=BRAIN_CONFIG["lookahead_sync"],
            slow_step_size=BRAIN_CONFIG["lookahead_alpha"],
        )
        self.ema = ParameterEMA(self.model, decay=BRAIN_CONFIG["ema_decay"])
        self._lock = threading.RLock()
        self.policy_strikes = 0
        self.policy_last_reset = time.time()

    @property
    def device(self) -> torch.device:
        return FLOAT_POLICY["device"]

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

    def _apply_hebbian_updates(self, reinforcement: float) -> Dict[str, Any]:
        applied = 0
        decay_rates: List[float] = []
        for layer in self.model.hebbian_layers():
            result = layer.update_hebbian(reinforcement)
            if result.get("applied"):
                applied += 1
            if "decay_rate" in result:
                decay_rates.append(float(result["decay_rate"]))
        return {"layers": len(self.model.hebbian_layers()), "applied": applied, "decay_rates": decay_rates}

    def _tensor_from_array(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array.astype(NP_FLOAT, copy=False)).to(self.device)
        return tensor

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
                with torch.no_grad():
                    ema_applied = False
                    try:
                        if hasattr(self, "ema") and self.ema is not None:
                            self.ema.apply_shadow(self.model)
                            ema_applied = True
                        tensor = torch.from_numpy(batch).to(self.device)
                        probs = self.model(tensor).cpu().numpy().astype(NP_FLOAT, copy=False)
                    finally:
                        if ema_applied:
                            self.ema.restore(self.model)
                for entry, row in zip(valid_entries, probs):
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
                    results[entry["index"]] = {
                        "action": action_index,
                        "weights_ok": bool(weights_ok),
                        "sanitized": {
                            "observation": entry["meta"],
                            "policy": policy_meta,
                            "exploration": bool(exploration),
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
                probs = self.model(obs_tensor)
                log_probs = torch.log(probs + 1e-8)
                selected_log_probs = log_probs.gather(1, action_tensor.view(-1, 1)).squeeze(1)
                loss = -(selected_log_probs * reward_tensor).mean()
                if not torch.isfinite(loss):
                    LOGGER.warning("Loss became non-finite; skipping update")
                else:
                    loss.backward()
                    params, dropped, clipped_grads, gradient_norm = self._filter_gradients()
                    weights_ok = True
                    weight_meta = {"sanitized": False, "replaced": 0, "clipped": 0}
                    if params:
                        self.optimizer.step()
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
                        dropped = max(dropped, 1)
                        clipped_grads = clipped_grads
                        gradient_norm = None
                        self.optimizer.zero_grad()
                        weights_ok = self._weights_are_finite()
                    mean_reward = float(np.mean([entry["reward"] for entry in valid_entries])) if valid_entries else 0.0
                    hebbian_info = self._apply_hebbian_updates(mean_reward)
                    for entry in valid_entries:
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
                            "hebbian": hebbian_info,
                        }
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
