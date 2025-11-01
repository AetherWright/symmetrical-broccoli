import atexit
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

ALLOWED_SEGMENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.")
STORAGE_ROOT = os.environ.get("TF_SERVER_STORAGE_ROOT") or os.path.join(os.getcwd(), "tf_server_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)


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

app = Flask(__name__)

BRAINS = {}
STATE_LOCK = threading.Lock()
STATUS = {
    "started_at": time.time(),
    "total_requests": 0,
    "requests": [],
    "bots": {},
    "workers": {}
}


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


def _sanitize_vector(vector, expected_size=None, label="vector"):
    if vector is None:
        return None, 0, 0, False
    try:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        app.logger.warning("Failed to coerce %s payload into float array", label)
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
        app.logger.warning(
            "Sanitized %s payload (replaced=%d, clipped=%d, adjusted=%s)",
            label,
            replaced,
            clipped,
            adjusted,
        )
    return arr.astype(np.float32, copy=False), replaced, clipped, adjusted


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
        activation="relu",
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
        dtype = self.dtype or tf.float32
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
        activation="relu",
        name="hidden_dense_1",
    )(x)
    x = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="hidden_dropout_1"
    )(x)
    hebbian_layers = []
    hebbian_1 = HebbianDense(
        BRAIN_CONFIG["hebbian_units"][0],
        activation="relu",
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
        activation="relu",
        name="hidden_dense_2",
    )(x)
    x = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="hidden_dropout_2"
    )(x)
    hebbian_2 = HebbianDense(
        BRAIN_CONFIG["hebbian_units"][1],
        activation="relu",
        hebbian_learning_rate=BRAIN_CONFIG["hebbian_learning_rate"],
        decay_multiplier=BRAIN_CONFIG["hebbian_decay_multiplier"],
        clip=BRAIN_CONFIG["hebbian_clip"],
        name="hebbian_dense_2",
    )
    x = hebbian_2(x)
    hebbian_layers.append(hebbian_2)
    shared = tf.keras.layers.Dense(
        max(1, BRAIN_CONFIG["shared_units"]),
        activation="relu",
        name="shared_dense",
    )(x)
    shared = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="shared_dropout"
    )(shared)
    policy_features = tf.keras.layers.Dense(
        max(1, BRAIN_CONFIG["shared_units"]),
        activation="relu",
        name="policy_dense",
    )(shared)
    policy_features = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="policy_features"
    )(policy_features)
    action_head = tf.keras.layers.Dense(
        action_count, activation="softmax", name="action_head"
    )(policy_features)
    prediction_features = tf.keras.layers.Dense(
        max(input_size, BRAIN_CONFIG["mid_units"]),
        activation="relu",
        name="prediction_dense",
    )(shared)
    prediction_head = tf.keras.layers.Dense(
        input_size, activation="linear", name="prediction_head"
    )(prediction_features)
    model = tf.keras.Model(inputs=inputs, outputs=[action_head, prediction_head])
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
        self.prediction_weight = tf.constant(1.0, dtype=tf.float32)
        self._lock = threading.RLock()

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

    def _sanitize_prediction_output(self, tensor):
        array = np.asarray(tensor, dtype=np.float32).reshape(-1)
        finite_mask = np.isfinite(array)
        replaced = int(array.size - np.count_nonzero(finite_mask))
        if replaced:
            app.logger.warning("Sanitized prediction output due to non-finite values")
        cleaned = np.nan_to_num(array, nan=0.0, posinf=OBS_CLAMP, neginf=-OBS_CLAMP)
        clipped = 0
        if OBS_CLAMP > 0:
            clipped_array = np.clip(cleaned, -OBS_CLAMP, OBS_CLAMP)
            clipped = int(np.count_nonzero(clipped_array != cleaned))
        else:
            clipped_array = cleaned
        adjusted = False
        if clipped_array.size != self.input_size:
            adjusted = True
            app.logger.warning(
                "Adjusted prediction output size from %d to %d",
                clipped_array.size,
                self.input_size,
            )
            if clipped_array.size > self.input_size:
                clipped_array = clipped_array[: self.input_size]
            else:
                clipped_array = np.pad(
                    clipped_array, (0, self.input_size - clipped_array.size), constant_values=0.0
                )
        metadata = {
            "replaced": replaced,
            "clipped": clipped,
            "adjusted": adjusted,
        }
        metadata["sanitized"] = bool(replaced or clipped or adjusted)
        return clipped_array.astype(np.float32, copy=False).tolist(), metadata

    @staticmethod
    def _filter_gradients(grads_and_vars):
        cleaned = []
        dropped = 0
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            tensor = grad.values if isinstance(grad, tf.IndexedSlices) else grad
            finite = bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
            if not finite:
                dropped += 1
                app.logger.warning("Dropped non-finite gradients for variable %s", getattr(var, "name", "?"))
                continue
            cleaned.append((grad, var))
        return cleaned, dropped

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

    def choose_action(self, observation, epsilon):
        with self._lock:
            try:
                obs, obs_meta = self._prepare_observation(observation, "observation")
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            action_probs, prediction = self.model(obs, training=False)
            raw_probs = action_probs.numpy().astype(np.float32, copy=False).reshape(-1)
            replaced = int(raw_probs.size - np.count_nonzero(np.isfinite(raw_probs)))
            sanitized_policy = replaced > 0
            if sanitized_policy:
                app.logger.warning("Sanitized action probabilities due to non-finite values")
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
                safe_probs = np.full(self.action_count, 1.0 / max(1, self.action_count), dtype=np.float32)
            else:
                safe_probs = safe_probs / total
            if not np.all(np.isfinite(safe_probs)):
                sanitized_policy = True
                safe_probs = np.nan_to_num(safe_probs, nan=1.0 / max(1, self.action_count))
            policy_meta = {
                "replaced": int(replaced),
                "fallback": bool(sanitized_policy),
            }
            predicted_next, prediction_meta = self._sanitize_prediction_output(prediction.numpy())
            weights_ok = policy_meta["replaced"] == 0 and prediction_meta["replaced"] == 0
            rand = float(np.random.random())
            exploration = rand < epsilon
            if exploration:
                action_index = int(np.random.randint(0, self.action_count))
            else:
                action_index = int(np.argmax(safe_probs))
            return {
                "action": action_index,
                "prediction": predicted_next,
                "weights_ok": bool(weights_ok),
                "sanitized": {
                    "observation": obs_meta,
                    "policy": policy_meta,
                    "prediction": prediction_meta,
                    "exploration": bool(exploration),
                },
            }

    def train(self, observation, action_index, reward, next_observation):
        with self._lock:
            try:
                obs, obs_meta = self._prepare_observation(observation, "observation")
            except ValueError:
                app.logger.warning("Skipping train call due to missing observation payload")
                return False

            use_policy = action_index is not None and reward is not None
            use_prediction = next_observation is not None

            if not (use_policy or use_prediction):
                return False

            bounded_action = None
            scaled_reward = 0.0
            if use_policy:
                try:
                    bounded_action = int(action_index)
                except (TypeError, ValueError):
                    app.logger.warning("Received invalid action index: %s", action_index)
                    use_policy = False
                else:
                    bounded_action = int(np.clip(bounded_action, 0, self.action_count - 1))
                    try:
                        reward_value = float(reward)
                    except (TypeError, ValueError):
                        app.logger.warning("Received invalid reward value: %s", reward)
                        reward_value = 0.0
                    if not np.isfinite(reward_value):
                        app.logger.warning("Reward contained non-finite value: %s", reward)
                        reward_value = 0.0
                    scaled_reward = float(np.clip(reward_value, -1.0, 1.0))

            next_obs = None
            next_meta = None
            if use_prediction:
                try:
                    next_obs, next_meta = self._prepare_observation(
                        next_observation, "next_observation"
                    )
                except ValueError:
                    app.logger.warning("Skipping prediction loss due to invalid next observation payload")
                    use_prediction = False

            with tf.GradientTape() as tape:
                action_pred, prediction = self.model(obs, training=True)
                total_loss = tf.constant(0.0, dtype=tf.float32)

                if use_policy:
                    one_hot = tf.one_hot([bounded_action], self.action_count)
                    log_probs = tf.math.log(action_pred + 1e-8)
                    policy_loss = -scaled_reward * tf.reduce_mean(
                        tf.reduce_sum(log_probs * one_hot, axis=-1)
                    )
                    total_loss += policy_loss

                if use_prediction:
                    prediction_loss = tf.reduce_mean(tf.square(prediction - next_obs))
                    total_loss += self.prediction_weight * prediction_loss

                total_loss = tf.where(
                    tf.math.is_finite(total_loss),
                    total_loss,
                    tf.constant(0.0, dtype=tf.float32),
                )

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            grads_and_vars = [
                (grad, var)
                for grad, var in zip(gradients, self.model.trainable_variables)
                if grad is not None
            ]
            cleaned_grads, dropped = self._filter_gradients(grads_and_vars)
            if dropped:
                app.logger.warning("Skipped %d gradient tensors due to non-finite values", dropped)
            weights_ok = True
            if cleaned_grads:
                self.optimizer.apply_gradients(cleaned_grads)
                weights_ok = self._weights_are_finite()
                if not weights_ok:
                    app.logger.error("Model weights contain non-finite values after training step")

            hebbian_info = self._apply_hebbian_updates(scaled_reward if use_policy else 0.0)

            return {
                "trained": bool(cleaned_grads),
                "weights_ok": bool(weights_ok),
                "dropped_gradients": int(dropped),
                "learning_rate": self._current_learning_rate(),
                "sanitized": {
                    "observation": obs_meta,
                    "next_observation": next_meta
                    if next_meta is not None
                    else {
                        "replaced": 0,
                        "clipped": 0,
                        "adjusted": False,
                        "sanitized": False,
                    },
                },
                "hebbian": hebbian_info,
            }

    def copy_from(self, other):
        if other is self:
            return
        first, second = (self, other) if id(self) <= id(other) else (other, self)
        with first._lock:
            with second._lock:
                weights = other.model.get_weights()
                self.model.set_weights(weights)
                self.optimizer.sync_slow_variables(self.model.trainable_variables)

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
            self.model.set_weights(averaged)
            self.optimizer.sync_slow_variables(self.model.trainable_variables)

    def mutate(self, stddev):
        with self._lock:
            weights = self.model.get_weights()
            mutated = []
            for weight in weights:
                noise = np.random.normal(0, stddev, size=weight.shape).astype(weight.dtype, copy=False)
                mutated_weight = (weight + noise).astype(weight.dtype, copy=False)
                mutated.append(mutated_weight)
            self.model.set_weights(mutated)
            self.optimizer.sync_slow_variables(self.model.trainable_variables)

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
                            app.logger.exception(
                                "Failed assigning partial weight %s for layer %s", weight_name, layer.name
                            )
                return loaded_variables, attempted_variables
        except Exception:  # pylint: disable=broad-except
            app.logger.exception("Partial weight load failed for %s", weights_path)
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
                app.logger.info("Restored weights exactly from %s", weights_path)
            except ValueError as exc:
                app.logger.warning(
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
                    app.logger.info("Restored compatible weights with skip_mismatch from %s", weights_path)
                except (TypeError, ValueError) as relaxed_exc:
                    app.logger.info(
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
                        app.logger.info(
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
                        app.logger.warning(
                            "No compatible tensors found when loading %s; keeping initialized weights",
                            weights_path,
                        )
            if loaded_info["status"] != "fresh":
                self.optimizer.sync_slow_variables(self.model.trainable_variables)
            else:
                app.logger.info("No compatible weights found at %s; using fresh initialization", weights_path)
            return loaded_info


def require_brain(brain_id):
    brain = BRAINS.get(brain_id)
    if brain is None:
        return None, (jsonify({"error": "Unknown brain"}), 404)
    return brain, None


def _execute(endpoint, func, *args, **kwargs):
    try:
        return WORKERS.run(endpoint, func, *args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.exception("Endpoint '%s' task failed", endpoint)
        raise RuntimeError(f"Failed to process {endpoint} request") from exc


def log_request(bot_id, endpoint, payload):
    with STATE_LOCK:
        STATUS["total_requests"] += 1
        entry = {
            "timestamp": time.time(),
            "bot_id": bot_id,
            "endpoint": endpoint,
            "reward": payload.get("reward"),
            "action": payload.get("action"),
            "epsilon": payload.get("epsilon"),
            "prediction": payload.get("prediction")
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
                    "epsilon": payload.get("epsilon")
                }
            )
            if endpoint == "act":
                bot_state["last_prediction"] = payload.get("prediction")
            elif endpoint == "train":
                next_obs = payload.get("next_observation")
                last_pred = bot_state.get("last_prediction")
                if (
                    isinstance(next_obs, (list, tuple))
                    and isinstance(last_pred, (list, tuple))
                    and len(next_obs) == len(last_pred)
                    and len(next_obs) > 0
                ):
                    diff = np.subtract(next_obs, last_pred)
                    mse = float(np.mean(np.square(diff)))
                    bot_state["prediction_mse"] = mse


@app.post("/api/brains")
def create_brain_endpoint():
    payload = request.get_json(force=True) or {}
    input_size = int(payload.get("input_size", 0))
    action_count = int(payload.get("action_count", 0))
    if input_size <= 0 or action_count <= 0:
        return jsonify({"error": "Invalid brain dimensions"}), 400
    brain_id = str(uuid.uuid4())
    BRAINS[brain_id] = RemoteBrain(input_size, action_count)
    return jsonify({"brain_id": brain_id})


@app.post("/api/brains/<brain_id>/act")
def choose_action_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    observation = payload.get("observation")
    if not isinstance(observation, (list, tuple)):
        return jsonify({"error": "Observation must be a list"}), 400
    epsilon = float(payload.get("epsilon", 0.1))
    try:
        result = _execute("act", brain.choose_action, observation, epsilon)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    if isinstance(result, dict):
        response_payload = {
            "action": int(result.get("action", 0)),
            "prediction": result.get("prediction"),
            "weights_ok": bool(result.get("weights_ok", True)),
            "sanitized": result.get("sanitized") or {},
        }
    else:
        action, prediction = result
        response_payload = {
            "action": int(action),
            "prediction": prediction,
            "weights_ok": True,
            "sanitized": {},
        }
    log_request(payload.get("bot_id"), "act", {**payload, **response_payload})
    return jsonify(response_payload)


@app.post("/api/brains/<brain_id>/train")
def train_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    observation = payload.get("observation")
    if not isinstance(observation, (list, tuple)):
        return jsonify({"error": "Observation must be a list"}), 400
    next_observation = payload.get("next_observation")
    try:
        result = _execute(
            "train",
            brain.train,
            observation,
            payload.get("action"),
            payload.get("reward"),
            next_observation if isinstance(next_observation, (list, tuple)) else None,
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
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
    return jsonify(response_payload)


@app.post("/api/brains/<brain_id>/copy")
def copy_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    source_id = payload.get("source_id")
    source, source_error = require_brain(source_id)
    if source_error:
        return source_error
    try:
        _execute("copy", brain.copy_from, source)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"status": "ok"})


@app.post("/api/brains/<brain_id>/average")
def average_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    source_ids = payload.get("source_ids") or []
    sources = []
    for source_id in source_ids:
        src, src_error = require_brain(source_id)
        if src_error:
            return src_error
        sources.append(src)
    try:
        _execute("average", brain.average_from, sources)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"status": "ok"})


@app.post("/api/brains/<brain_id>/mutate")
def mutate_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    stddev = float(payload.get("stddev", 0.02))
    if stddev <= 0:
        return jsonify({"error": "Stddev must be positive"}), 400
    try:
        _execute("mutate", brain.mutate, stddev)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"status": "ok"})


@app.post("/api/brains/<brain_id>/save")
def save_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    label = payload.get("path") or brain_id
    target_dir = _resolve_storage_path("brains", label, create=True)
    try:
        _execute("save", brain.save, target_dir)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"status": "ok", "path": label})


@app.post("/api/brains/load")
def load_endpoint():
    payload = request.get_json(force=True) or {}
    path_label = payload.get("path")
    input_size = int(payload.get("input_size", 0))
    action_count = int(payload.get("action_count", 0))
    if not path_label or input_size <= 0 or action_count <= 0:
        return jsonify({"error": "Invalid load request"}), 400
    path = _resolve_storage_path("brains", path_label, create=False)
    if not os.path.isdir(path):
        return jsonify({"error": "Checkpoint not found"}), 404
    brain_id = str(uuid.uuid4())
    brain = RemoteBrain(input_size, action_count)
    try:
        load_result = _execute("load", brain.load_weights, path)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    BRAINS[brain_id] = brain
    response = {"brain_id": brain_id}
    if isinstance(load_result, dict):
        response.update(
            {key: value for key, value in load_result.items() if key not in {"path"}}
        )
        if load_result.get("status") in {"exact", "relaxed", "partial"}:
            response["source_path"] = load_result.get("path")
    return jsonify(response)


@app.post("/api/state/save")
def save_state_endpoint():
    payload = request.get_json(force=True) or {}
    label = payload.get("path")
    if not label:
        return jsonify({"error": "Path is required"}), 400
    state = payload.get("state") or {}
    path = _resolve_storage_path("state", label, create=True)
    state_path = os.path.join(path, "brain_state.json")
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return jsonify({"status": "ok", "path": label})


@app.post("/api/state/load")
def load_state_endpoint():
    payload = request.get_json(force=True) or {}
    label = payload.get("path")
    if not label:
        return jsonify({"error": "Path is required"}), 400
    path = _resolve_storage_path("state", label, create=False)
    state_path = os.path.join(path, "brain_state.json")
    if not os.path.exists(state_path):
        return jsonify({"state": None})
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    return jsonify({"state": state})


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
            "workers": STATUS["workers"]
        }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
