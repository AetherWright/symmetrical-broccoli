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
    "hidden_units": 64,
    "dropout_rate": 0.2
}


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


def build_model(input_size, action_count):
    inputs = tf.keras.Input(shape=(input_size,), name="observation")
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(BRAIN_CONFIG["hidden_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(BRAIN_CONFIG["dropout_rate"])(x)
    shared = tf.keras.layers.Dense(
        max(1, BRAIN_CONFIG["hidden_units"] // 2),
        activation="relu",
        name="shared_dense"
    )(x)
    policy_features = tf.keras.layers.Dropout(
        BRAIN_CONFIG["dropout_rate"], name="policy_features"
    )(shared)
    action_head = tf.keras.layers.Dense(
        action_count, activation="softmax", name="action_head"
    )(policy_features)
    prediction_head = tf.keras.layers.Dense(
        input_size, activation="linear", name="prediction_head"
    )(policy_features)
    model = tf.keras.Model(inputs=inputs, outputs=[action_head, prediction_head])
    return model


class RemoteBrain:
    def __init__(self, input_size, action_count):
        self.input_size = int(input_size)
        self.action_count = int(action_count)
        self.model = build_model(self.input_size, self.action_count)
        nadam = tf.keras.optimizers.Nadam(learning_rate=2e-3)
        self.optimizer = LookaheadOptimizer(nadam, sync_period=6, slow_step_size=0.5)
        self.prediction_weight = tf.constant(1.0, dtype=tf.float32)
        self._lock = threading.RLock()

    def choose_action(self, observation, epsilon):
        with self._lock:
            obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
            action_probs, prediction = self.model(obs, training=False)
            probs = np.clip(action_probs.numpy().flatten(), 1e-8, 1.0)
            probs = probs / probs.sum()
            predicted_next = prediction.numpy().flatten().tolist()
            if np.random.random() < epsilon:
                action_index = int(np.random.randint(0, self.action_count))
            else:
                action_index = int(np.argmax(probs))
            return action_index, predicted_next

    def train(self, observation, action_index, reward, next_observation):
        with self._lock:
            obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)

            use_policy = action_index is not None and reward is not None
            use_prediction = next_observation is not None

            if not (use_policy or use_prediction):
                return False

            with tf.GradientTape() as tape:
                action_pred, prediction = self.model(obs, training=True)
                total_loss = tf.constant(0.0, dtype=tf.float32)

                if use_policy:
                    bounded_action = int(np.clip(int(action_index), 0, self.action_count - 1))
                    scaled_reward = float(np.clip(reward, -1.0, 1.0))
                    one_hot = tf.one_hot([bounded_action], self.action_count)
                    log_probs = tf.math.log(action_pred + 1e-8)
                    policy_loss = -scaled_reward * tf.reduce_mean(
                        tf.reduce_sum(log_probs * one_hot, axis=-1)
                    )
                    total_loss += policy_loss

                if use_prediction:
                    next_obs = np.asarray(next_observation, dtype=np.float32).reshape(1, -1)
                    prediction_loss = tf.reduce_mean(tf.square(prediction - next_obs))
                    total_loss += self.prediction_weight * prediction_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            grads_and_vars = [
                (grad, var)
                for grad, var in zip(gradients, self.model.trainable_variables)
                if grad is not None
            ]
            if grads_and_vars:
                self.optimizer.apply_gradients(grads_and_vars)

            return bool(grads_and_vars)

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

    def load_weights(self, directory):
        with self._lock:
            weights_path = os.path.join(directory, "weights.weights.h5")
            if not os.path.exists(weights_path):
                legacy_path = os.path.join(directory, "weights.h5")
                if os.path.exists(legacy_path):
                    weights_path = legacy_path
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                self.optimizer.sync_slow_variables(self.model.trainable_variables)


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
        action, prediction = _execute("act", brain.choose_action, observation, epsilon)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    log_request(
        payload.get("bot_id"),
        "act",
        {**payload, "action": action, "prediction": prediction}
    )
    return jsonify({"action": action, "prediction": prediction})


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
        trained = _execute(
            "train",
            brain.train,
            observation,
            payload.get("action"),
            payload.get("reward"),
            next_observation if isinstance(next_observation, (list, tuple)) else None
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    log_request(payload.get("bot_id"), "train", payload)
    return jsonify({"trained": bool(trained)})


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
        _execute("load", brain.load_weights, path)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    BRAINS[brain_id] = brain
    return jsonify({"brain_id": brain_id})


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
