import atexit
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from fastapi import Body, FastAPI, HTTPException
import uvicorn

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

ALLOWED_SEGMENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.")
STORAGE_ROOT = os.environ.get("TF_SERVER_STORAGE_ROOT") or os.path.join(os.getcwd(), "tf_server_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)

LOGGER = logging.getLogger("tf_server")


def _configure_tensorflow_devices():
    info = {
        "available": False,
        "physical": 0,
        "logical": 0,
        "memory_growth": 0,
        "names": [],
    }
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Unable to inspect GPU devices: %s", exc)
        return info
    if not gpus:
        LOGGER.info("No GPU devices detected; TensorFlow will run on CPU.")
        return info
    info["available"] = True
    info["physical"] = len(gpus)
    info["names"] = [getattr(device, "name", str(device)) for device in gpus]
    configured = 0
    for device in gpus:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            configured += 1
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to enable memory growth for %s: %s", device, exc)
    info["memory_growth"] = configured
    try:
        logical = tf.config.list_logical_devices("GPU")
        info["logical"] = len(logical)
        LOGGER.info(
            "TensorFlow GPU acceleration detected (%d physical, %d logical). Memory growth configured on %d device(s).",
            info["physical"],
            info["logical"],
            configured,
        )
    except Exception:  # pragma: no cover - defensive
        LOGGER.info(
            "TensorFlow GPU acceleration detected (%d physical GPU devices). Memory growth configured on %d device(s).",
            info["physical"],
            configured,
        )
    return info


def _configure_float_policy(gpu_info):
    requested_gpu = str(os.environ.get("TF_SERVER_ENABLE_GPU", "0")).lower() in {"1", "true", "yes", "on"}
    using_gpu = bool(gpu_info.get("available") and requested_gpu)
    if not using_gpu and gpu_info.get("available"):
        try:
            tf.config.set_visible_devices([], "GPU")
            LOGGER.info("GPU devices disabled; running on CPU with float64 policy.")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to disable GPU devices: %s", exc)
    floatx = "float32" if using_gpu else "float64"
    try:
        tf.keras.backend.set_floatx(floatx)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to set TensorFlow floatx to %s: %s", floatx, exc)
    if using_gpu:
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
        except Exception:  # pragma: no cover - best effort
            pass
    return {
        "floatx": floatx,
        "tf": tf.float32 if floatx == "float32" else tf.float64,
        "np": np.float32 if floatx == "float32" else np.float64,
        "using_gpu": using_gpu,
    }


def _sanitize_segment(value, fallback="default"):
    if not value:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    sanitized = "".join(ch for ch in text if ch in ALLOWED_SEGMENT_CHARS)
    return sanitized or fallback


def _resolve_storage_path(*segments, create=False):
    parts = [_sanitize_segment(seg) for seg in segments if seg is not None]
    directory = os.path.join(STORAGE_ROOT, *parts)
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory

app = FastAPI()

BRAINS = {}
STATE_LOCK = threading.Lock()


def natural_log_relu(inputs):
    tensor = tf.convert_to_tensor(inputs)
    return tf.math.log1p(tf.nn.relu(tensor))


tf.keras.utils.get_custom_objects()["natural_log_relu"] = natural_log_relu


def _read_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _read_float(name, default):
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
    "default": DEFAULT_WORKERS
}

ACT_SANITIZATION_STRIKE_WINDOW = max(1.0, _read_float("TF_SERVER_ACT_STRIKE_WINDOW", 30.0))
ACT_SANITIZATION_STRIKE_THRESHOLD = max(1, _read_int("TF_SERVER_ACT_STRIKE_THRESHOLD", 2))
ACT_SANITIZATION_FORCE_THRESHOLD = max(
    ACT_SANITIZATION_STRIKE_THRESHOLD,
    _read_int("TF_SERVER_ACT_FORCE_THRESHOLD", 4),
)

STATUS = {
    "started_at": time.time(),
    "total_requests": 0,
    "requests": [],
    "bots": {},
    "workers": {}
}

GPU_INFO = _configure_tensorflow_devices()
STATUS["accelerators"] = {"gpu": GPU_INFO}

FLOAT_POLICY = _configure_float_policy(GPU_INFO)
TF_FLOAT = FLOAT_POLICY["tf"]
NP_FLOAT = FLOAT_POLICY["np"]
STATUS["float_policy"] = {
    "floatx": FLOAT_POLICY["floatx"],
    "using_gpu": FLOAT_POLICY["using_gpu"],
}
STATUS["policy_guard"] = {
    "act_strike_window": ACT_SANITIZATION_STRIKE_WINDOW,
    "act_strike_threshold": ACT_SANITIZATION_STRIKE_THRESHOLD,
    "act_force_threshold": ACT_SANITIZATION_FORCE_THRESHOLD,
}
LOGGER.info(
    "TensorFlow float policy set to %s (using_gpu=%s).",
    FLOAT_POLICY["floatx"],
    FLOAT_POLICY["using_gpu"],
)


class EndpointWorkerPool:
    def __init__(self, limits):
        self._limits = dict(limits)
        self._executors = {}
        self._lock = threading.Lock()

    def _get_limit(self, endpoint):
        return max(1, int(self._limits.get(endpoint, self._limits.get("default", 1))))

    def _get_executor(self, endpoint):
        with self._lock:
            executor = self._executors.get(endpoint)
            if executor is None:
                max_workers = self._get_limit(endpoint)
                executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"{endpoint}-worker"
                )
                self._executors[endpoint] = executor
            return executor

    def run(self, endpoint, func, *args, **kwargs):
        executor = self._get_executor(endpoint)
        future = executor.submit(func, *args, **kwargs)
        return future.result()

    def snapshot(self):
        snapshot = {}
        with self._lock:
            for endpoint, executor in self._executors.items():
                queue = getattr(executor, "_work_queue", None)
                pending = queue.qsize() if queue is not None else 0
                snapshot[endpoint] = {
                    "max_workers": executor._max_workers,
                    "pending": pending
                }
        return snapshot

    def shutdown(self):
        with self._lock:
            executors = list(self._executors.values())
            self._executors.clear()
        for executor in executors:
            executor.shutdown(wait=False)


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
    "learning_rate": 2e-3,
    "cosine_first_decay_steps": 1024,
    "cosine_t_mul": 2.0,
    "cosine_m_mul": 1.0,
    "cosine_alpha": 0.0,
    "lion_beta_1": 0.9,
    "lion_beta_2": 0.99,
    "lion_ema_momentum": 0.99,
}

OBS_CLAMP = abs(_read_float("TF_SERVER_OBSERVATION_CLAMP", 1e6))
GRAD_CLIP_VALUE = abs(_read_float("TF_SERVER_GRADIENT_CLIP_VALUE", 100.0))
GRAD_CLIP_GLOBAL_NORM = abs(_read_float("TF_SERVER_GRADIENT_GLOBAL_NORM", 250.0))
GRAD_SKIP_GLOBAL_NORM = abs(_read_float("TF_SERVER_GRADIENT_SKIP_GLOBAL_NORM", 0.0))
WEIGHT_CLAMP = abs(_read_float("TF_SERVER_WEIGHT_CLAMP", 1e6))


def _sanitize_vector(vector, expected_size=None, label="vector"):
    if vector is None:
        return None, 0, 0, False
    try:
        arr = np.asarray(vector, dtype=NP_FLOAT).reshape(-1)
    except (TypeError, ValueError):
        LOGGER.warning("Failed to coerce %s payload into float array", label)
        return None, 0, 0, False
    invalid_mask = ~np.isfinite(arr)
    replaced = int(np.count_nonzero(invalid_mask))
    if replaced:
        arr = np.where(invalid_mask, 0.0, arr)
    clipped = 0
    if OBS_CLAMP > 0:
        clipped_mask = np.abs(arr) > OBS_CLAMP
        clipped = int(np.count_nonzero(clipped_mask))
        if clipped:
            arr = np.clip(arr, -OBS_CLAMP, OBS_CLAMP)
    adjusted = False
    if expected_size and expected_size > 0 and arr.size != expected_size:
        adjusted = True
        if arr.size > expected_size:
            arr = arr[:expected_size]
        else:
            arr = np.pad(arr, (0, expected_size - arr.size), constant_values=0.0)
    if replaced or clipped or adjusted:
        LOGGER.warning(
            "Sanitized %s payload (replaced=%d, clipped=%d, adjusted=%s)",
            label,
            replaced,
            clipped,
            adjusted,
        )
    return arr.astype(NP_FLOAT, copy=False), replaced, clipped, adjusted


class LookaheadOptimizer:
    def __init__(self, optimizer, sync_period=6, slow_step_size=0.5):
        self.optimizer = optimizer
        self.sync_period = max(1, int(sync_period))
        self.slow_step_size = float(slow_step_size)
        self._fast_vars = []
        self._slow_vars = []
        self._fast_var_ids = {}
        self._step = 0

    @staticmethod
    def _read_variable(variable):
        value_attr = getattr(variable, "value", None)
        if value_attr is not None:
            if callable(value_attr):
                return value_attr()
            try:
                return tf.convert_to_tensor(value_attr)
            except TypeError:
                pass
        if hasattr(variable, "read_value"):
            return variable.read_value()
        if hasattr(variable, "numpy"):
            return tf.convert_to_tensor(variable.numpy())
        return tf.convert_to_tensor(variable)

    def _ensure_slot_variables(self, variables):
        for var in variables:
            var_id = id(var)
            if var_id in self._fast_var_ids:
                continue
            self._fast_vars.append(var)
            initial_value = self._read_variable(var)
            var_dtype = getattr(var, "dtype", None)
            if var_dtype is not None:
                slow_var = tf.Variable(initial_value, dtype=var_dtype, trainable=False)
            else:
                slow_var = tf.Variable(initial_value, trainable=False)
            self._slow_vars.append(slow_var)
            self._fast_var_ids[var_id] = len(self._fast_vars) - 1

    def sync_slow_variables(self, variables=None):
        if variables is None:
            variables = list(self._fast_vars)
        else:
            self._ensure_slot_variables(variables)
        for var in variables:
            index = self._fast_var_ids.get(id(var))
            if index is None:
                continue
            fast_value = self._read_variable(self._fast_vars[index])
            self._slow_vars[index].assign(fast_value)

    def get_slow_variable_value(self, variable):
        index = self._fast_var_ids.get(id(variable))
        if index is None:
            return None
        slow_var = self._slow_vars[index]
        try:
            return slow_var.numpy()
        except AttributeError:
            return np.asarray(slow_var)

    def apply_gradients(self, grads_and_vars):
        if not grads_and_vars:
            return
        variables = [var for grad, var in grads_and_vars if var is not None]
        if not variables:
            return
        self._ensure_slot_variables(variables)
        self.optimizer.apply_gradients(grads_and_vars)
        self._step += 1
        if self.sync_period and self._step % self.sync_period == 0:
            for slow_var, fast_var in zip(self._slow_vars, self._fast_vars):
                fast_value = self._read_variable(fast_var)
                slow_value = self._read_variable(slow_var)
                slow_var.assign(slow_value + (fast_value - slow_value) * self.slow_step_size)
                fast_var.assign(self._read_variable(slow_var))


class HebbianDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=natural_log_relu,
        hebbian_learning_rate=0.01,
        decay_multiplier=1.5,
        clip=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.hebbian_learning_rate = float(abs(hebbian_learning_rate))
        self.decay_multiplier = float(abs(decay_multiplier))
        self.clip = float(abs(clip))
        self._last_input = None
        self._last_output = None

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        dtype = self.dtype or TF_FLOAT
        self.base_kernel = self.add_weight(
            name="base_kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            dtype=dtype,
        )
        self.base_bias = self.add_weight(
            name="base_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            dtype=dtype,
        )
        self.hebbian_kernel = self.add_weight(
            name="hebbian_kernel",
            shape=(last_dim, self.units),
            initializer="zeros",
            trainable=False,
            dtype=dtype,
        )
        self.hebbian_bias = self.add_weight(
            name="hebbian_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=False,
            dtype=dtype,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        tensor = tf.cast(inputs, self.base_kernel.dtype)
        combined_kernel = self.base_kernel + self.hebbian_kernel
        combined_bias = self.base_bias + self.hebbian_bias
        outputs = tf.linalg.matmul(tensor, combined_kernel)
        outputs = tf.nn.bias_add(outputs, combined_bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if training:
            self._last_input = tf.stop_gradient(tf.identity(tensor))
            self._last_output = tf.stop_gradient(tf.identity(outputs))
        return outputs

    def _apply_decay(self):
        if self.hebbian_learning_rate <= 0:
            return 0.0
        decay_rate = self.hebbian_learning_rate * self.decay_multiplier
        decay_rate = min(0.95, max(0.0, decay_rate))
        if decay_rate == 0:
            return 0.0
        keep_ratio = 1.0 - decay_rate
        keep_ratio_tensor = tf.cast(keep_ratio, self.hebbian_kernel.dtype)
        self.hebbian_kernel.assign(self.hebbian_kernel * keep_ratio_tensor)
        self.hebbian_bias.assign(self.hebbian_bias * keep_ratio_tensor)
        return float(decay_rate)

    def update_hebbian(self, reinforcement):
        decay_rate = self._apply_decay()
        reinforcement = float(np.clip(reinforcement, -1.0, 1.0))
        applied = False
        if reinforcement != 0.0 and self._last_input is not None and self._last_output is not None:
            lr = self.hebbian_learning_rate * reinforcement
            pre = tf.cast(self._last_input, self.hebbian_kernel.dtype)
            post = tf.cast(self._last_output, self.hebbian_kernel.dtype)
            pre = tf.nn.l2_normalize(pre, axis=-1)
            post = tf.nn.l2_normalize(post, axis=-1)
            kernel_update = tf.einsum("bi,bj->ij", pre, post)
            kernel_update /= tf.cast(tf.shape(pre)[0], self.hebbian_kernel.dtype)
            bias_update = tf.reduce_mean(post, axis=0)
            step = tf.cast(lr, self.hebbian_kernel.dtype)
            self.hebbian_kernel.assign_add(kernel_update * step)
            self.hebbian_bias.assign_add(bias_update * step)
            applied = True
        if self.clip > 0:
            clip_value = tf.cast(self.clip, self.hebbian_kernel.dtype)
            self.hebbian_kernel.assign(
                tf.clip_by_value(self.hebbian_kernel, -clip_value, clip_value)
            )
            self.hebbian_bias.assign(
                tf.clip_by_value(self.hebbian_bias, -clip_value, clip_value)
            )
        self._last_input = None
        self._last_output = None
        return {"applied": applied, "decay_rate": decay_rate}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "hebbian_learning_rate": self.hebbian_learning_rate,
                "decay_multiplier": self.decay_multiplier,
                "clip": self.clip,
            }
        )
        return config


tf.keras.utils.get_custom_objects()["HebbianDense"] = HebbianDense


def build_model(input_size, action_count):
    inputs = tf.keras.Input(shape=(input_size,), name="observation")
    x = tf.keras.layers.BatchNormalization(name="input_batchnorm")(inputs)
    x = tf.keras.layers.Dense(
        BRAIN_CONFIG["hidden_units"],
        activation=natural_log_relu,
        name="hidden_dense_1",
    )(x)
    x = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="hidden_dropout_1"
    )(x)
    hebbian_layers = []
    hebbian_1 = HebbianDense(
        BRAIN_CONFIG["hebbian_units"][0],
        activation=natural_log_relu,
        hebbian_learning_rate=BRAIN_CONFIG["hebbian_learning_rate"],
        decay_multiplier=BRAIN_CONFIG["hebbian_decay_multiplier"],
        clip=BRAIN_CONFIG["hebbian_clip"],
        name="hebbian_dense_1",
    )
    x = hebbian_1(x)
    hebbian_layers.append(hebbian_1)
    x = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="hebbian_dropout_1"
    )(x)
    x = tf.keras.layers.Dense(
        BRAIN_CONFIG["mid_units"],
        activation=natural_log_relu,
        name="hidden_dense_2",
    )(x)
    x = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="hidden_dropout_2"
    )(x)
    hebbian_2 = HebbianDense(
        BRAIN_CONFIG["hebbian_units"][1],
        activation=natural_log_relu,
        hebbian_learning_rate=BRAIN_CONFIG["hebbian_learning_rate"],
        decay_multiplier=BRAIN_CONFIG["hebbian_decay_multiplier"],
        clip=BRAIN_CONFIG["hebbian_clip"],
        name="hebbian_dense_2",
    )
    x = hebbian_2(x)
    hebbian_layers.append(hebbian_2)
    shared = tf.keras.layers.Dense(
        max(1, BRAIN_CONFIG["shared_units"]),
        activation=natural_log_relu,
        name="shared_dense",
    )(x)
    shared = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="shared_dropout"
    )(shared)
    policy_features = tf.keras.layers.Dense(
        max(1, BRAIN_CONFIG["shared_units"]),
        activation=natural_log_relu,
        name="policy_dense",
    )(shared)
    policy_features = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="policy_features"
    )(policy_features)
    action_head = tf.keras.layers.Dense(
        action_count, activation="softmax", name="action_head"
    )(policy_features)
    model = tf.keras.Model(inputs=inputs, outputs=action_head)
    model._hebbian_layers = hebbian_layers  # type: ignore[attr-defined]
    return model


class RemoteBrain:
    def __init__(self, input_size, action_count):
        self.input_size = int(input_size)
        self.action_count = int(action_count)
        self.model = build_model(self.input_size, self.action_count)
        self.hebbian_layers = list(getattr(self.model, "_hebbian_layers", []))
        first_decay_steps = max(1, int(BRAIN_CONFIG["cosine_first_decay_steps"]))
        learning_rate = float(BRAIN_CONFIG["learning_rate"])
        cosine_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=float(BRAIN_CONFIG["cosine_t_mul"]),
            m_mul=float(BRAIN_CONFIG["cosine_m_mul"]),
            alpha=float(BRAIN_CONFIG["cosine_alpha"]),
        )
        lion = tf.keras.optimizers.Lion(
            learning_rate=cosine_schedule,
            beta_1=float(BRAIN_CONFIG["lion_beta_1"]),
            beta_2=float(BRAIN_CONFIG["lion_beta_2"]),
            use_ema=True,
            ema_momentum=float(BRAIN_CONFIG["lion_ema_momentum"]),
        )
        self.optimizer = LookaheadOptimizer(lion, sync_period=6, slow_step_size=0.5)
        self.learning_rate_schedule = cosine_schedule
        self._lock = threading.RLock()
        self.policy_strikes = 0
        self.policy_last_reset = time.time()

    def _prepare_observation(self, vector, label):
        prepared, replaced, clipped, adjusted = _sanitize_vector(
            vector, expected_size=self.input_size, label=label
        )
        if prepared is None:
            raise ValueError(f"{label} payload missing")
        metadata = {
            "replaced": int(replaced),
            "clipped": int(clipped),
            "adjusted": bool(adjusted),
        }
        metadata["sanitized"] = bool(replaced or clipped or adjusted)
        return prepared.reshape(1, -1), metadata

    def _current_learning_rate(self):
        base_optimizer = getattr(self.optimizer, "optimizer", None)
        if base_optimizer is None:
            return None
        learning_rate = getattr(base_optimizer, "learning_rate", None)
        if learning_rate is None:
            return None
        try:
            if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                iterations = getattr(base_optimizer, "iterations", None)
                step = iterations if iterations is not None else tf.constant(0, dtype=tf.int64)
                value = learning_rate(step)
            else:
                value = learning_rate
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            elif hasattr(value, "numpy"):
                value = value.numpy()
            return float(value)
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _filter_gradients(grads_and_vars):
        gradients = []
        variables = []
        dropped = 0
        clipped = 0
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            tensor = grad.values if isinstance(grad, tf.IndexedSlices) else grad
            finite = bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
            if not finite:
                dropped += 1
                LOGGER.warning(
                    "Dropped non-finite gradients for variable %s",
                    getattr(var, "name", "?"),
                )
                continue
            clipped_grad = grad
            if GRAD_CLIP_VALUE > 0:
                clip_limit = tf.cast(GRAD_CLIP_VALUE, dtype=tensor.dtype)
                exceeds = bool(
                    tf.reduce_any(tf.math.greater(tf.math.abs(tensor), clip_limit)).numpy()
                )
                if exceeds:
                    clipped_tensor = tf.clip_by_value(tensor, -clip_limit, clip_limit)
                    if isinstance(grad, tf.IndexedSlices):
                        clipped_grad = tf.IndexedSlices(
                            clipped_tensor, grad.indices, grad.dense_shape
                        )
                    else:
                        clipped_grad = clipped_tensor
                    clipped += 1
            gradients.append(clipped_grad)
            variables.append(var)
        global_norm = None
        if gradients:
            if GRAD_CLIP_GLOBAL_NORM > 0:
                try:
                    clipped_list, original_norm = tf.clip_by_global_norm(
                        gradients, GRAD_CLIP_GLOBAL_NORM
                    )
                    gradients = clipped_list
                    try:
                        norm_value = original_norm.numpy()
                    except AttributeError:
                        norm_value = float(original_norm)
                    global_norm = float(norm_value)
                    if global_norm > GRAD_CLIP_GLOBAL_NORM:
                        clipped += 1
                        LOGGER.debug(
                            "Clipped gradients by global norm (norm=%.3f limit=%.3f)",
                            global_norm,
                            GRAD_CLIP_GLOBAL_NORM,
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to apply global norm clipping: %s", exc)
                    gradients = list(gradients)
            else:
                try:
                    norm_tensor = tf.linalg.global_norm(
                        [
                            grad.values if isinstance(grad, tf.IndexedSlices) else grad
                            for grad in gradients
                        ]
                    )
                    global_norm = float(norm_tensor.numpy())
                except Exception:  # pragma: no cover - monitoring best effort
                    global_norm = None
        cleaned = list(zip(gradients, variables))
        if global_norm is not None:
            if not np.isfinite(global_norm):
                LOGGER.warning(
                    "Global gradient norm became non-finite; dropping %d gradient tensors",
                    len(cleaned),
                )
                return [], dropped + len(cleaned), clipped, None
            if GRAD_SKIP_GLOBAL_NORM > 0 and global_norm > GRAD_SKIP_GLOBAL_NORM:
                LOGGER.warning(
                    "Global gradient norm %.3f exceeded skip threshold %.3f; skipping gradient application",
                    global_norm,
                    GRAD_SKIP_GLOBAL_NORM,
                )
                return [], dropped + len(cleaned), clipped, global_norm
        return cleaned, dropped, clipped, global_norm

    @staticmethod
    def _sanitize_weight_list(weights):
        sanitized_weights = []
        replaced_total = 0
        clipped_total = 0
        sanitized_any = False
        for weight in weights:
            array = np.asarray(weight)
            dtype = getattr(weight, "dtype", array.dtype)
            invalid_mask = ~np.isfinite(array)
            invalid_count = int(np.count_nonzero(invalid_mask))
            clip_count = 0
            clip_needed = False
            if WEIGHT_CLAMP > 0:
                clip_needed = bool(np.any(np.abs(array) > WEIGHT_CLAMP))
            if invalid_count or clip_needed:
                sanitized = array.astype(dtype, copy=True)
                if invalid_count:
                    sanitized[invalid_mask] = 0.0
                if WEIGHT_CLAMP > 0:
                    clip_mask = np.abs(sanitized) > WEIGHT_CLAMP
                    clip_count = int(np.count_nonzero(clip_mask))
                    if clip_count:
                        np.clip(sanitized, -WEIGHT_CLAMP, WEIGHT_CLAMP, out=sanitized)
                sanitized_weights.append(sanitized.astype(dtype, copy=False))
                sanitized_any = True
            else:
                sanitized_weights.append(array.astype(dtype, copy=False))
            replaced_total += invalid_count
            clipped_total += clip_count
        return sanitized_weights, {
            "sanitized": bool(sanitized_any),
            "replaced": int(replaced_total),
            "clipped": int(clipped_total),
        }

    def _assign_weights(self, weights, reason):
        sanitized_weights, meta = self._sanitize_weight_list(weights)
        final_weights = []
        for original, sanitized in zip(weights, sanitized_weights):
            array = np.asarray(sanitized)
            dtype = getattr(original, "dtype", array.dtype)
            if array.dtype != dtype:
                array = array.astype(dtype, copy=False)
            final_weights.append(array)
        self.model.set_weights(final_weights)
        self.optimizer.sync_slow_variables(self.model.trainable_variables)
        if meta["sanitized"]:
            LOGGER.warning(
                "Sanitized %s weights (replaced=%d, clipped=%d)",
                reason,
                meta["replaced"],
                meta["clipped"],
            )
        return meta

    def _sanitize_model_weights(self, reason):
        replaced_total = 0
        clipped_total = 0
        sanitized_any = False
        slow_value_lookup = None
        fetch_slow_value = getattr(self.optimizer, "get_slow_variable_value", None)
        if callable(fetch_slow_value):
            slow_value_lookup = fetch_slow_value
        trainable_ids = {id(var): var for var in self.model.trainable_variables}
        for variable in self.model.weights:
            array = variable.numpy()
            dtype = getattr(variable.dtype, "as_numpy_dtype", array.dtype)
            invalid_mask = ~np.isfinite(array)
            invalid_count = int(np.count_nonzero(invalid_mask))
            finite_array = np.nan_to_num(
                array,
                nan=0.0,
                posinf=WEIGHT_CLAMP + 1.0 if WEIGHT_CLAMP > 0 else 0.0,
                neginf=-(WEIGHT_CLAMP + 1.0) if WEIGHT_CLAMP > 0 else 0.0,
            )
            clip_needed = bool(WEIGHT_CLAMP > 0 and np.any(np.abs(finite_array) > WEIGHT_CLAMP))
            if not invalid_count and not clip_needed:
                continue
            sanitized_any = True
            sanitized = array.astype(dtype, copy=True)
            if invalid_count:
                replacement = None
                if slow_value_lookup is not None and id(variable) in trainable_ids:
                    slow_value = slow_value_lookup(trainable_ids[id(variable)])
                    if slow_value is not None and np.shape(slow_value) == sanitized.shape:
                        replacement = np.asarray(slow_value, dtype=dtype)
                        if not np.all(np.isfinite(replacement)):
                            replacement = np.nan_to_num(replacement, nan=0.0, posinf=0.0, neginf=0.0)
                if replacement is not None:
                    sanitized[invalid_mask] = replacement[invalid_mask]
                else:
                    sanitized[invalid_mask] = 0.0
                replaced_total += invalid_count
            clip_count = 0
            if WEIGHT_CLAMP > 0:
                clip_mask = np.abs(sanitized) > WEIGHT_CLAMP
                clip_count = int(np.count_nonzero(clip_mask))
                if clip_count:
                    np.clip(sanitized, -WEIGHT_CLAMP, WEIGHT_CLAMP, out=sanitized)
                    clipped_total += clip_count
            if sanitized.dtype != dtype:
                sanitized = sanitized.astype(dtype, copy=False)
            variable.assign(sanitized)
        meta = {
            "sanitized": bool(sanitized_any),
            "replaced": int(replaced_total),
            "clipped": int(clipped_total),
        }
        if meta["sanitized"]:
            self.optimizer.sync_slow_variables(self.model.trainable_variables)
            LOGGER.warning(
                "Sanitized %s weights (replaced=%d, clipped=%d)",
                reason,
                meta["replaced"],
                meta["clipped"],
            )
        return meta

    def _weights_are_finite(self):
        for weight in self.model.get_weights():
            if not np.all(np.isfinite(weight)):
                return False
        return True

    def _apply_hebbian_updates(self, reinforcement):
        if not self.hebbian_layers:
            return {"layers": 0, "applied": 0, "decay_rates": []}
        applied = 0
        decay_rates = []
        for layer in self.hebbian_layers:
            result = layer.update_hebbian(reinforcement)
            if not isinstance(result, dict):
                continue
            decay_rate = result.get("decay_rate")
            if decay_rate is not None:
                decay_rates.append(float(decay_rate))
            if result.get("applied"):
                applied += 1
        return {
            "layers": len(self.hebbian_layers),
            "applied": applied,
            "decay_rates": decay_rates,
        }

    def choose_actions_batch(self, observations, epsilons):
        with self._lock:
            count = len(observations)
            results = [None] * count
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
                        "tensor": obs.reshape(-1),
                        "meta": obs_meta,
                    }
                )
            if valid_entries:
                batch = np.stack([entry["tensor"] for entry in valid_entries], axis=0).astype(
                    NP_FLOAT, copy=False
                )
                action_probs = self.model(batch, training=False)
                probs_np = action_probs.numpy().astype(NP_FLOAT, copy=False)
                for entry, row in zip(valid_entries, probs_np):
                    raw_probs = row.astype(NP_FLOAT, copy=False).reshape(-1)
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
                        safe_probs = np.full(
                            self.action_count,
                            1.0 / max(1, self.action_count),
                            dtype=NP_FLOAT,
                        )
                    else:
                        safe_probs = safe_probs / total
                    if not np.all(np.isfinite(safe_probs)):
                        sanitized_policy = True
                        safe_probs = np.nan_to_num(
                            safe_probs,
                            nan=1.0 / max(1, self.action_count),
                        )
                    now = time.time()
                    weights_ok = replaced == 0
                    strikes = None
                    if sanitized_policy:
                        LOGGER.warning(
                            "Sanitized action probabilities due to non-finite values"
                        )
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
                    }
                    policy_meta["sanitized"] = bool(
                        policy_meta["replaced"] or policy_meta["fallback"]
                    )
                    if strikes is not None:
                        policy_meta["strikes"] = int(strikes)
                        policy_meta["strike_threshold"] = int(
                            ACT_SANITIZATION_FORCE_THRESHOLD
                        )
                        policy_meta["window_ms"] = int(
                            ACT_SANITIZATION_STRIKE_WINDOW * 1000
                        )
                    results[entry["index"]] = {
                        "action": action_index,
                        "weights_ok": bool(weights_ok),
                        "sanitized": {
                            "observation": entry["meta"],
                            "policy": policy_meta,
                            "exploration": bool(exploration),
                        },
                    }
            for index, value in enumerate(results):
                if value is None:
                    results[index] = {"error": "Observation could not be processed"}
            return results

    def choose_action(self, observation, epsilon):
        batch = self.choose_actions_batch([observation], [epsilon])
        if not batch:
            raise RuntimeError("Failed to compute action")
        result = batch[0]
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def train_batch(
        self, observations, actions, rewards, penalties, next_observations
    ):
        with self._lock:
            count = len(observations)
            results = [None] * count
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
                            "observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "next_observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "weights": {
                                "replaced": 0,
                                "clipped": 0,
                                "sanitized": False,
                            },
                        },
                    }
                    continue

                bounded_action = None
                reward_value = rewards[index] if index < len(rewards) else None
                penalty_value = penalties[index] if index < len(penalties) else None
                action_value = actions[index] if index < len(actions) else None
                if action_value is None or (
                    reward_value is None and penalty_value is None
                ):
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "sanitized": {
                            "observation": obs_meta,
                            "next_observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "weights": {
                                "replaced": 0,
                                "clipped": 0,
                                "sanitized": False,
                            },
                        },
                    }
                    continue
                try:
                    bounded_action = int(action_value)
                except (TypeError, ValueError):
                    LOGGER.warning("Received invalid action index: %s", action_value)
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "sanitized": {
                            "observation": obs_meta,
                            "next_observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "weights": {
                                "replaced": 0,
                                "clipped": 0,
                                "sanitized": False,
                            },
                        },
                    }
                    continue
                bounded_action = int(np.clip(bounded_action, 0, self.action_count - 1))
                positive_component = _sanitize_component(
                    reward_value, "reward component"
                )
                penalty_component = _sanitize_component(
                    penalty_value, "penalty component"
                )

                reward_float = positive_component - penalty_component
                log_reward_component = float(np.log1p(positive_component))
                log_penalty_component = float(np.log1p(penalty_component))
                scaled_reward = log_reward_component - log_penalty_component
                if not np.isfinite(scaled_reward):
                    LOGGER.warning(
                        "Reward transformation produced non-finite value (raw=%s, log_reward=%s, log_penalty=%s)",
                        reward_float,
                        log_reward_component,
                        log_penalty_component,
                    )
                    scaled_reward = 0.0

                next_meta = {
                    "replaced": 0,
                    "clipped": 0,
                    "adjusted": False,
                    "sanitized": False,
                }
                if index < len(next_observations):
                    next_observation = next_observations[index]
                    if next_observation is not None:
                        try:
                            _, next_meta = self._prepare_observation(
                                next_observation, "next_observation"
                            )
                        except ValueError:
                            LOGGER.warning(
                                "Skipping next observation sanitization due to invalid payload"
                            )

                valid_entries.append(
                    {
                        "index": index,
                        "obs": obs.reshape(-1),
                        "obs_meta": obs_meta,
                        "next_meta": next_meta,
                        "action": bounded_action,
                        "reward": scaled_reward,
                        "reward_meta": {
                            "raw": reward_float,
                            "positive": positive_component,
                            "penalty": penalty_component,
                            "log_positive": log_reward_component,
                            "log_penalty": log_penalty_component,
                        },
                    }
                )

            if valid_entries:
                obs_matrix = np.stack([entry["obs"] for entry in valid_entries], axis=0).astype(
                    NP_FLOAT, copy=False
                )
                actions_tensor = tf.convert_to_tensor(
                    [entry["action"] for entry in valid_entries], dtype=tf.int32
                )
                rewards_tensor = tf.convert_to_tensor(
                    [entry["reward"] for entry in valid_entries], dtype=TF_FLOAT
                )

                with tf.GradientTape() as tape:
                    action_pred = self.model(obs_matrix, training=True)
                    action_pred = tf.cast(action_pred, TF_FLOAT)
                    total_loss = tf.constant(0.0, dtype=TF_FLOAT)
                    one_hot = tf.one_hot(
                        actions_tensor, self.action_count, dtype=TF_FLOAT
                    )
                    log_probs = tf.math.log(
                        action_pred + tf.constant(1e-8, dtype=TF_FLOAT)
                    )
                    policy_loss = -tf.reduce_mean(
                        tf.reduce_sum(log_probs * one_hot, axis=-1) * rewards_tensor
                    )
                    total_loss += policy_loss
                    total_loss = tf.where(
                        tf.math.is_finite(total_loss),
                        total_loss,
                        tf.constant(0.0, dtype=TF_FLOAT),
                    )

                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                grads_and_vars = [
                    (grad, var)
                    for grad, var in zip(gradients, self.model.trainable_variables)
                    if grad is not None
                ]
                cleaned_grads, dropped, clipped_grads, gradient_norm = self._filter_gradients(
                    grads_and_vars
                )
                if dropped:
                    LOGGER.warning(
                        "Skipped %d gradient tensors due to non-finite values", dropped
                    )
                weights_ok = True
                weight_meta = {
                    "sanitized": False,
                    "replaced": 0,
                    "clipped": 0,
                    "reason": "post-train-batch",
                }
                if cleaned_grads:
                    self.optimizer.apply_gradients(cleaned_grads)
                    weights_ok = self._weights_are_finite()
                    if not weights_ok:
                        LOGGER.error(
                            "Model weights contain non-finite values after training step"
                        )
                    weight_meta = self._sanitize_model_weights("post-train-batch")
                    if weight_meta.get("sanitized"):
                        if weight_meta.get("replaced"):
                            weights_ok = False
                        else:
                            weights_ok = weights_ok and self._weights_are_finite()
                    else:
                        weights_ok = weights_ok and self._weights_are_finite()

                mean_reward = float(np.mean([entry["reward"] for entry in valid_entries]))
                hebbian_info = self._apply_hebbian_updates(mean_reward)
                current_lr = self._current_learning_rate()

                for entry in valid_entries:
                    reward_meta = entry.get("reward_meta", {})
                    results[entry["index"]] = {
                        "trained": bool(cleaned_grads),
                        "weights_ok": bool(weights_ok),
                        "dropped_gradients": int(dropped),
                        "clipped_gradients": int(clipped_grads),
                        "gradient_norm": float(gradient_norm) if gradient_norm is not None else None,
                        "learning_rate": current_lr,
                        "sanitized": {
                            "observation": entry["obs_meta"],
                            "next_observation": entry["next_meta"],
                            "weights": {
                                "replaced": int(weight_meta.get("replaced", 0)),
                                "clipped": int(weight_meta.get("clipped", 0)),
                                "sanitized": bool(weight_meta.get("sanitized", False)),
                            },
                            "reward": reward_meta,
                        },
                        "hebbian": hebbian_info,
                    }

            for index, value in enumerate(results):
                if value is None:
                    results[index] = {
                        "trained": False,
                        "weights_ok": True,
                        "dropped_gradients": 0,
                        "clipped_gradients": 0,
                        "gradient_norm": None,
                        "sanitized": {
                            "observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "next_observation": {
                                "replaced": 0,
                                "clipped": 0,
                                "adjusted": False,
                                "sanitized": False,
                            },
                            "weights": {
                                "replaced": 0,
                                "clipped": 0,
                                "sanitized": False,
                            },
                        },
                    }
            return results

    def train(self, observation, action_index, reward, penalty, next_observation):
        batch = self.train_batch(
            [observation],
            [action_index],
            [reward],
            [penalty],
            [next_observation],
        )
        if not batch:
            return False
        return batch[0]

    def copy_from(self, other):
        if other is self:
            return
        first, second = (self, other) if id(self) <= id(other) else (other, self)
        with first._lock:
            with second._lock:
                weights = other.model.get_weights()
                self._assign_weights(weights, "copy")

    def average_from(self, sources):
        if not sources:
            return
        with self._lock:
            weights = []
            for source in sources:
                if source is None:
                    continue
                with source._lock:
                    weights.append(source.model.get_weights())
            if not weights:
                return
            averaged = []
            for layer_weights in zip(*weights):
                layer_stack = np.stack(layer_weights, axis=0)
                averaged_layer = layer_stack.mean(axis=0)
                if averaged_layer.dtype != layer_stack.dtype:
                    averaged_layer = averaged_layer.astype(layer_stack.dtype)
                averaged.append(averaged_layer)
            self._assign_weights(averaged, "average")

    def mutate(self, stddev):
        with self._lock:
            weights = self.model.get_weights()
            mutated = []
            for weight in weights:
                noise = np.random.normal(0, stddev, size=weight.shape).astype(weight.dtype, copy=False)
                mutated_weight = (weight + noise).astype(weight.dtype, copy=False)
                mutated.append(mutated_weight)
            self._assign_weights(mutated, "mutate")

    def save(self, directory):
        with self._lock:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "model.json"), "w", encoding="utf-8") as handle:
                handle.write(self.model.to_json())
            weights_path = os.path.join(directory, "weights.weights.h5")
            legacy_path = os.path.join(directory, "weights.h5")
            if os.path.exists(legacy_path) and legacy_path != weights_path:
                try:
                    os.remove(legacy_path)
                except OSError:
                    pass
            self.model.save_weights(weights_path)

    @staticmethod
    def _decode_name(name):
        if isinstance(name, (bytes, bytearray)):
            return name.decode("utf-8")
        return str(name)

    def _load_partial_weights(self, weights_path):
        if h5py is None or not os.path.exists(weights_path):
            return 0, 0
        loaded_variables = 0
        attempted_variables = 0
        try:
            with h5py.File(weights_path, "r") as handle:  # type: ignore[call-arg]
                root_group = handle
                if "model_weights" in handle:
                    root_group = handle["model_weights"]
                layer_names = root_group.attrs.get("layer_names")
                if isinstance(layer_names, np.ndarray):
                    layer_names = layer_names.tolist()
                if not layer_names:
                    return 0, 0
                layer_lookup = {}
                for raw_name in layer_names:
                    layer_name = self._decode_name(raw_name)
                    if layer_name in root_group:
                        layer_lookup[layer_name] = root_group[layer_name]
                if not layer_lookup:
                    return 0, 0
                for layer in self.model.layers:
                    group = layer_lookup.get(layer.name)
                    if group is None:
                        continue
                    weight_names = group.attrs.get("weight_names")
                    if isinstance(weight_names, np.ndarray):
                        weight_names = weight_names.tolist()
                    if not weight_names:
                        continue
                    for index, weight_var in enumerate(layer.weights):
                        attempted_variables += 1
                        try:
                            raw_weight_name = weight_names[index]
                        except IndexError:
                            break
                        weight_name = self._decode_name(raw_weight_name)
                        if weight_name not in group:
                            continue
                        value = group[weight_name][()]
                        target_shape = tuple(int(dim) for dim in weight_var.shape)
                        if value.shape != target_shape:
                            continue
                        if (
                            hasattr(weight_var, "dtype")
                            and hasattr(weight_var.dtype, "as_numpy_dtype")
                        ):
                            value = value.astype(weight_var.dtype.as_numpy_dtype, copy=False)
                        try:
                            weight_var.assign(value)
                            loaded_variables += 1
                        except Exception:  # pylint: disable=broad-except
                            LOGGER.exception(
                                "Failed assigning partial weight %s for layer %s", weight_name, layer.name
                            )
                return loaded_variables, attempted_variables
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Partial weight load failed for %s", weights_path)
            return loaded_variables, attempted_variables

    def load_weights(self, directory):
        with self._lock:
            weights_path = os.path.join(directory, "weights.weights.h5")
            if not os.path.exists(weights_path):
                legacy_path = os.path.join(directory, "weights.h5")
                if os.path.exists(legacy_path):
                    weights_path = legacy_path
            if not os.path.exists(weights_path):
                return {"status": "fresh"}

            loaded_info = {"status": "fresh"}
            try:
                self.model.load_weights(weights_path)
                loaded_info = {"status": "exact", "path": weights_path}
                weight_meta = self._sanitize_model_weights("load")
                if weight_meta.get("sanitized"):
                    loaded_info["weight_sanitized"] = {
                        "replaced": int(weight_meta.get("replaced", 0)),
                        "clipped": int(weight_meta.get("clipped", 0)),
                    }
                LOGGER.info("Restored weights exactly from %s", weights_path)
            except ValueError as exc:
                LOGGER.warning(
                    "Exact weight load failed for %s: %s. Attempting relaxed restore.",
                    weights_path,
                    exc,
                )
                relaxed_loaded = False
                try:
                    self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                    relaxed_loaded = True
                    loaded_info = {
                        "status": "relaxed",
                        "path": weights_path,
                    }
                    weight_meta = self._sanitize_model_weights("load-relaxed")
                    if weight_meta.get("sanitized"):
                        loaded_info["weight_sanitized"] = {
                            "replaced": int(weight_meta.get("replaced", 0)),
                            "clipped": int(weight_meta.get("clipped", 0)),
                        }
                    LOGGER.info("Restored compatible weights with skip_mismatch from %s", weights_path)
                except (TypeError, ValueError) as relaxed_exc:
                    LOGGER.info(
                        "Relaxed weight load unavailable for %s: %s",
                        weights_path,
                        relaxed_exc,
                    )
                if not relaxed_loaded:
                    loaded_vars, attempted = self._load_partial_weights(weights_path)
                    if loaded_vars:
                        loaded_info = {
                            "status": "partial",
                            "path": weights_path,
                            "loaded_variables": loaded_vars,
                            "attempted_variables": attempted,
                        }
                        weight_meta = self._sanitize_model_weights("load-partial")
                        if weight_meta.get("sanitized"):
                            loaded_info["weight_sanitized"] = {
                                "replaced": int(weight_meta.get("replaced", 0)),
                                "clipped": int(weight_meta.get("clipped", 0)),
                            }
                        LOGGER.info(
                            "Partially restored %s/%s variables from %s",
                            loaded_vars,
                            attempted or "?",
                            weights_path,
                        )
                    else:
                        loaded_info = {
                            "status": "mismatch",
                            "path": weights_path,
                            "loaded_variables": loaded_vars,
                            "attempted_variables": attempted,
                        }
                        LOGGER.warning(
                            "No compatible tensors found when loading %s; keeping initialized weights",
                            weights_path,
                        )
            if loaded_info["status"] != "fresh":
                self.optimizer.sync_slow_variables(self.model.trainable_variables)
            else:
                LOGGER.info("No compatible weights found at %s; using fresh initialization", weights_path)
            return loaded_info


def require_brain(brain_id):
    brain = BRAINS.get(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Unknown brain")
    return brain


def _execute(endpoint, func, *args, **kwargs):
    try:
        return WORKERS.run(endpoint, func, *args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Endpoint '%s' task failed", endpoint)
        raise RuntimeError(f"Failed to process {endpoint} request") from exc


def log_request(bot_id, endpoint, payload):
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
                if isinstance(policy_meta, dict):
                    if "strikes" in policy_meta:
                        try:
                            bot_state["policy_strikes"] = int(policy_meta["strikes"])
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            bot_state["policy_strikes"] = policy_meta["strikes"]
                        bot_state["policy_strike_threshold"] = ACT_SANITIZATION_FORCE_THRESHOLD
                    if policy_meta.get("sanitized"):
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
def choose_action_endpoint(
    brain_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)
):
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
    if isinstance(result, dict):
        response_payload = {
            "action": int(result.get("action", 0)),
            "weights_ok": bool(result.get("weights_ok", True)),
            "sanitized": result.get("sanitized") or {},
        }
    else:
        response_payload = {
            "action": int(result),
            "weights_ok": True,
            "sanitized": {},
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
    if isinstance(result, dict):
        response_payload = {
            "trained": bool(result.get("trained")),
            "weights_ok": bool(result.get("weights_ok", True)),
            "dropped_gradients": int(result.get("dropped_gradients", 0)),
            "sanitized": result.get("sanitized") or {},
        }
    else:
        response_payload = {"trained": bool(result), "weights_ok": True}
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
                results[index] = {
                    "trained": False,
                    "weights_ok": True,
                    "dropped_gradients": 0,
                    "clipped_gradients": 0,
                    "gradient_norm": None,
                    "sanitized": {
                        "observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                        "next_observation": {"replaced": 0, "clipped": 0, "adjusted": False, "sanitized": False},
                        "weights": {"replaced": 0, "clipped": 0, "sanitized": False},
                    },
                }
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
            results[index] = {
                "trained": bool(entry_result.get("trained")),
                "weights_ok": bool(entry_result.get("weights_ok", True)),
                "dropped_gradients": int(entry_result.get("dropped_gradients", 0)),
                "clipped_gradients": int(entry_result.get("clipped_gradients", 0)),
                "gradient_norm": entry_result.get("gradient_norm"),
                "learning_rate": entry_result.get("learning_rate"),
                "hebbian": entry_result.get("hebbian"),
                "sanitized": entry_result.get("sanitized") or {},
            }
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
    sources = []
    for source_id in source_ids:
        sources.append(require_brain(source_id))
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
        if load_result.get("status") in {"exact", "relaxed", "partial"}:
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
            "bots": STATUS["bots"],
            "recent_requests": STATUS["requests"][-10:],
            "workers": STATUS["workers"],
        }
    return payload


if __name__ == "__main__":
    port = int(os.environ.get("TF_SERVER_PORT", 5000))
    uvicorn.run("tf_server:app", host="0.0.0.0", port=port, log_level="info")
