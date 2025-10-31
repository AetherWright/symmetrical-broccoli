import json
import os
import threading
import time
import uuid

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
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
    "bots": {}
}

BRAIN_CONFIG = {
    "hidden_units": 64,
    "dropout_rate": 0.2
}


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
        self.optimizer = tfa.optimizers.Lookahead(nadam, sync_period=6, slow_step_size=0.5)
        self.prediction_weight = tf.constant(1.0, dtype=tf.float32)

    def choose_action(self, observation, epsilon):
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
        self.model.set_weights(other.model.get_weights())

    def average_from(self, sources):
        if not sources:
            return
        weights = [source.model.get_weights() for source in sources]
        averaged = [np.mean(np.stack(layer_weights, axis=0), axis=0) for layer_weights in zip(*weights)]
        self.model.set_weights(averaged)

    def mutate(self, stddev):
        weights = self.model.get_weights()
        mutated = [w + np.random.normal(0, stddev, size=w.shape) for w in weights]
        self.model.set_weights(mutated)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "model.json"), "w", encoding="utf-8") as handle:
            handle.write(self.model.to_json())
        weights_path = os.path.join(directory, "weights.h5")
        self.model.save_weights(weights_path)

    def load_weights(self, directory):
        weights_path = os.path.join(directory, "weights.h5")
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)


def require_brain(brain_id):
    brain = BRAINS.get(brain_id)
    if brain is None:
        return None, (jsonify({"error": "Unknown brain"}), 404)
    return brain, None


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
    action, prediction = brain.choose_action(observation, epsilon)
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
    trained = brain.train(
        observation,
        payload.get("action"),
        payload.get("reward"),
        next_observation if isinstance(next_observation, (list, tuple)) else None
    )
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
    brain.copy_from(source)
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
    brain.average_from(sources)
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
    brain.mutate(stddev)
    return jsonify({"status": "ok"})


@app.post("/api/brains/<brain_id>/save")
def save_endpoint(brain_id):
    brain, error = require_brain(brain_id)
    if error:
        return error
    payload = request.get_json(force=True) or {}
    label = payload.get("path") or brain_id
    target_dir = _resolve_storage_path("brains", label, create=True)
    brain.save(target_dir)
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
    brain.load_weights(path)
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
        payload = {
            "started_at": STATUS["started_at"],
            "uptime": time.time() - STATUS["started_at"],
            "total_requests": STATUS["total_requests"],
            "brain_count": len(BRAINS),
            "bots": STATUS["bots"],
            "recent_requests": STATUS["requests"][-10:]
        }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
