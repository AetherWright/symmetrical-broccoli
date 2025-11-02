"""HTTP bridge for communicating with the Echo inference server."""
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Tuple

import requests
from requests import Response, Session

from .config import EchoConfig

LOGGER = logging.getLogger(__name__)


class EchoBridgeError(RuntimeError):
    """Raised when the Echo bridge cannot fulfil a request."""


class EchoBridge:
    """Bridge that talks to the Echo inference and learning API."""

    def __init__(
        self,
        config: Optional[EchoConfig] = None,
        *,
        session: Optional[Session] = None,
    ) -> None:
        self.config = config or EchoConfig()
        self._session = session or requests.Session()

    def infer_action(
        self,
        observations: object,
        actions: Optional[Sequence[object]] = None,
        *,
        temperature: Optional[float] = None,
        prefer_pipeline: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """Ask the Echo server to choose an action.

        Args:
            observations: Observation payload accepted by Echo.
            actions: Optional list of candidate actions.
            temperature: Optional override for sampling temperature.
            prefer_pipeline: Optional override for the prefer_pipeline flag.

        Returns:
            A tuple of (action, confidence).

        Raises:
            EchoBridgeError: If the request fails after retries or the server
                response is invalid.
        """

        payload = {
            "observations": observations,
        }
        if actions is not None:
            payload["actions"] = list(actions)
        if temperature is None:
            temperature = self.config.default_temperature
        payload["temperature"] = temperature
        if prefer_pipeline is None:
            prefer_pipeline = self.config.prefer_pipeline
        payload["prefer_pipeline"] = prefer_pipeline

        result = self._post_with_retries(self.config.infer_url, payload)
        if not isinstance(result, dict):
            raise EchoBridgeError("Invalid response type from Echo inference endpoint")

        action = result.get("action")
        confidence = result.get("confidence", 0.0)
        if not isinstance(action, str) or not action:
            raise EchoBridgeError("Echo inference response missing action")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0

        confidence_value = max(0.0, min(1.0, confidence_value))
        return action, confidence_value

    def learn(
        self,
        observation: object,
        next_observation: object,
        action: object,
        reward: float,
    ) -> None:
        """Send a learning update to the Echo server."""

        payload = {
            "observation": observation,
            "next_observation": next_observation,
            "action": action,
            "reward": reward,
        }
        self._post_with_retries(self.config.learn_url, payload)

    def _post_with_retries(self, url: str, payload: dict) -> dict:
        delay = self.config.retry_backoff_seconds
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self._session.post(url, json=payload, timeout=self.config.timeout_seconds)
                latency_ms = (time.perf_counter() - start_time) * 1000
                LOGGER.debug("Echo request to %s completed in %.2fms", url, latency_ms)
                response.raise_for_status()
                return self._parse_json(response)
            except (requests.RequestException, EchoBridgeError) as exc:
                last_exception = exc
                LOGGER.warning(
                    "Echo request to %s failed on attempt %s/%s: %s",
                    url,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
            except ValueError as exc:
                last_exception = exc
                LOGGER.warning(
                    "Echo request to %s returned invalid JSON on attempt %s/%s: %s",
                    url,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
            if attempt < self.config.max_retries:
                time.sleep(delay)
                delay *= 2
        raise EchoBridgeError(f"Echo request to {url} failed after retries") from last_exception

    @staticmethod
    def _parse_json(response: Response) -> dict:
        try:
            return response.json()
        except ValueError as exc:
            raise EchoBridgeError("Failed to decode JSON response from Echo server") from exc


__all__ = ["EchoBridge", "EchoBridgeError"]
