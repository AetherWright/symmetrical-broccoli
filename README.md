# Multi-Bot Minecraft Brain Stack

This stack contains three cooperating services that power a multi-bot Minecraft experiment:

- **`tf_server.py`** – a FastAPI service that hosts TensorFlow brains with custom optimizers, batching, sanitization, and storage helpers.
- **`bot.js`** – the Mineflayer controller that spawns a generation of bots, handles lifecycle management, and delegates action selection/training to the remote brain service through `brainClient.js` and the worker pool helpers.
- **`echo_integration/`** – a lightweight bridge for the optional Echo inference service, providing high-level wrappers for `/infer` and `/learn` HTTP calls.

## Prerequisites

1. **Python 3.10+** with a virtual environment for the TensorFlow service.
2. **Node.js 18+** (the Mineflayer stack depends on modern ES modules and worker threads).
3. A reachable Minecraft server (Java Edition) the bots can join – by default the controller connects to `localhost:25565` but you can edit `MC_HOST` and `MC_PORT` at the top of `bot.js` to match your world.
4. (Optional) Access to an Echo inference server exposing `/infer` and `/learn` endpoints if you plan to use the echo integration helpers.

## Installation

```bash
# Install Node dependencies
npm install

# Create & activate a Python virtual environment, then install server deps
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the TensorFlow brain server

The brain service runs on FastAPI with uvicorn. Start it before launching the bots:

```bash
source .venv/bin/activate
python tf_server.py
```

Environment flags you may want to tune:

- `TF_SERVER_PORT` – change the listening port (defaults to 5000).
- `TF_SERVER_STORAGE_ROOT` – directory for weights and saved state.
- `TF_SERVER_ENABLE_GPU` – set to `1`/`true` to allow GPU execution; when unset the server forces CPU + float64 for numeric stability.

The server exposes `/status` for health snapshots and `/api/brains/*` endpoints for brain lifecycle, batching, and storage.

## Running the bot controller

Start the bot swarm once the brain service is reachable and your Minecraft server is up:

```bash
node bot.js
```

Key runtime knobs come from environment variables (`process.env`) read in `bot.js` and `brainClient.js`, including:

- `BOT_COUNT`, `BOT_MIN`, `BOT_MAX` – size of the active population.
- `GENERATION_TICKS`, `CROSSOVER_INTERVAL` – evolution cadence controls.
- `BOT_AUTOEAT_THRESHOLD`, `BOT_HUNGER_CRITICAL`, `BOT_HUNT_HUNGER_THRESHOLD` – upkeep thresholds.
- `BOT_LOW_REWARD_THRESHOLD`, `BOT_DOWNTREND_WINDOW`, etc. – automated retirement heuristics.
- `TF_SERVER_URL`, `OBSERVATION_CLAMP` – remote brain endpoint and client-side sanitization clamp.

The worker pool in `brainWorkerPool.js` automatically batches inference/training calls, and bots tick at 20 Hz (50 ms interval) to stay in sync with the Minecraft server loop.

## Reward pipeline

### Client-side shaping

`bot.js` constructs dense rewards by combining many context-sensitive components. Movement, survival, resource gain, social cues, and behavioural diversity each contribute through `applyRewardComponent`, which enforces sign expectations, caps magnitude, and tracks separate positive/negative tallies before combining them into a final scalar.【F:bot.js†L355-L372】【F:bot.js†L4842-L5078】 The accumulator is finalized just before transmission so both the total reward and its decomposed components remain bounded.【F:bot.js†L330-L352】

### Server-side normalization

The FastAPI brain server receives the shaped reward and accompanying penalty tally. `_transform_reward` log-scales both positive reward and penalty components after sanitizing non-finite or negative inputs, then converts them into the single scalar used for optimisation while preserving detailed metadata for debugging.【F:tf_server.py†L1010-L1039】 During training, each batch report echoes the sanitized observation metadata and the reward decomposition to make downstream monitoring consistent with the client-side shaping.【F:tf_server.py†L1180-L1239】

### Learning approach

Training follows an advantage actor-critic style update: the policy head outputs action logits, while a parallel reward head predicts the expected scaled return for each action. The server computes advantages by subtracting the detached reward prediction baseline from the received reward, centres them, and applies a policy-gradient loss alongside a smooth L1 regression loss for the critic head.【F:tf_server.py†L1184-L1195】 This pairing gives the closest analogue to actor-critic reinforcement learning within the system while still leveraging the custom reward shaping described above.

## Echo integration (optional)

The `echo_integration` package lets you proxy observations and rewards to an external Echo inference service.

```python
from echo_integration.agent import EchoAgent

echo_agent = EchoAgent()
action, confidence = echo_agent.act(observation, actions)
echo_agent.learn(observation, next_observation, action, reward)
```

Configuration values can be provided through environment variables or by instantiating `EchoConfig` manually:

- `ECHO_INFER_URL`, `ECHO_LEARN_URL` – endpoint locations (default `http://localhost:8000`).
- `ECHO_TIMEOUT`, `ECHO_MAX_RETRIES`, `ECHO_RETRY_BACKOFF` – retry/backoff policy for HTTP calls.
- `ECHO_TEMPERATURE`, `ECHO_PREFER_PIPELINE` – inference sampling behaviour.
- `ECHO_AUTOSAVE_INTERVAL` – hint for consumers that persist Echo-related metadata.

## Operational tips

- Watch the bot logs for sanitization or recovery notices – persistent NaN/Inf warnings indicate the brain may need to rebuild from clean peers.
- Use the `/status` endpoint on the TensorFlow server to inspect worker pool metrics, active brains, and recent request history.
- When running overnight, keep `tmux`/`screen` sessions or a process manager (pm2/systemd) so both Node and Python services auto-restart if they exit.

## Development & Testing

- Static sanity check for the Python modules: `python -m compileall tf_server.py echo_integration`.
- Node syntax check: `node --check bot.js` (requires Node 20+).
- The repository has no default unit test suite; integrate with your preferred harness if you extend the stack.
