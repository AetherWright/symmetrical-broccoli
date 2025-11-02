# Symmetrical Broccoli Bot Stack

Symmetrical Broccoli contains three cooperating services that power a multi-bot Minecraft experiment:

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
