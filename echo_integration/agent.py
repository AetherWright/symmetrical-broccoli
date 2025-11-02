"""Agent wrapper that orchestrates interactions with the Echo bridge."""
from __future__ import annotations

import json
import logging
from collections import deque
from typing import Deque, Optional, Sequence, Tuple

from .bridge import EchoBridge, EchoBridgeError
from .config import EchoConfig

LOGGER = logging.getLogger(__name__)

class EchoAgent:
    """High-level helper that manages Echo interactions for an environment."""

    def __init__(
        self,
        bridge: Optional[EchoBridge] = None,
        *,
        config: Optional[EchoConfig] = None,
        reward_window: int = 100,
    ) -> None:
        self.config = config or EchoConfig()
        self.bridge = bridge or EchoBridge(self.config)
        self.reward_window: Deque[float] = deque(maxlen=reward_window)
        self.current_episode_reward = 0.0
        self.episode_count = 0
        self._last_observation: Optional[object] = None
        self._last_action: Optional[object] = None

    def reset_episode(self) -> None:
        """Reset episodic tracking, saving accumulated reward history."""

        self.reward_window.append(self.current_episode_reward)
        LOGGER.debug(
            "EchoAgent finishing episode %s with reward %.3f",
            self.episode_count,
            self.current_episode_reward,
        )
        self.current_episode_reward = 0.0
        self._last_observation = None
        self._last_action = None
        self.episode_count += 1

    def act(
        self,
        observation: object,
        actions: Optional[Sequence[object]] = None,
        *,
        temperature: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Request an action choice from the Echo inference server."""

        formatted_observation = self._format_observation(observation)
        formatted_actions = self._format_actions(actions)
        try:
            action, confidence = self.bridge.infer_action(
                formatted_observation,
                formatted_actions,
                temperature=temperature,
            )
        except EchoBridgeError:
            LOGGER.exception("EchoAgent failed to retrieve action from Echo server")
            raise

        self._last_observation = formatted_observation
        self._last_action = action
        return action, confidence

    def learn(
        self,
        observation: object,
        next_observation: object,
        action: object,
        reward: float,
        *,
        auto_track: bool = True,
    ) -> None:
        """Report a transition to the Echo learning endpoint."""

        formatted_observation = self._format_observation(observation)
        formatted_next_observation = self._format_observation(next_observation)
        formatted_action = self._format_action_value(action)
        try:
            self.bridge.learn(
                formatted_observation,
                formatted_next_observation,
                formatted_action,
                float(reward),
            )
        except EchoBridgeError:
            LOGGER.exception("EchoAgent failed to submit learning update to Echo server")
            raise

        if auto_track:
            self.current_episode_reward += float(reward)

    def _format_observation(self, observation: object) -> object:
        if observation is None:
            return None
        if isinstance(observation, (str, bytes)):
            return observation.decode("utf-8", errors="ignore") if isinstance(observation, bytes) else observation
        if isinstance(observation, (int, float, bool)):
            return observation
        if isinstance(observation, dict):
            return {str(key): self._format_observation(value) for key, value in observation.items()}
        if isinstance(observation, (list, tuple)):
            return [self._format_observation(value) for value in observation]
        try:
            return json.loads(json.dumps(observation))
        except (TypeError, ValueError):
            return str(observation)

    def _format_actions(self, actions: Optional[Sequence[object]]) -> Optional[Sequence[object]]:
        if actions is None:
            return None
        return [self._format_action_value(action) for action in actions]

    @staticmethod
    def _format_action_value(action: object) -> object:
        if isinstance(action, (str, int, float)):
            return action
        if isinstance(action, bool):
            return int(action)
        if isinstance(action, (list, tuple)):
            return [EchoAgent._format_action_value(value) for value in action]
        if isinstance(action, bytes):
            return action.decode("utf-8", errors="ignore")
        if isinstance(action, dict):
            return {str(key): EchoAgent._format_action_value(value) for key, value in action.items()}
        return str(action)


__all__ = ["EchoAgent"]
