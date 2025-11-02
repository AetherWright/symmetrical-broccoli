"""Configuration helpers for the Echo integration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


_DEF_INFER_URL = "http://localhost:8000/infer"
_DEF_LEARN_URL = "http://localhost:8000/learn"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class EchoConfig:
    """Runtime configuration for the Echo bridge."""

    infer_url: str = os.environ.get("ECHO_INFER_URL", _DEF_INFER_URL)
    learn_url: str = os.environ.get("ECHO_LEARN_URL", _DEF_LEARN_URL)
    timeout_seconds: float = _get_float("ECHO_TIMEOUT", 8.0)
    max_retries: int = _get_int("ECHO_MAX_RETRIES", 3)
    retry_backoff_seconds: float = _get_float("ECHO_RETRY_BACKOFF", 0.5)
    prefer_pipeline: bool = _get_bool("ECHO_PREFER_PIPELINE", True)
    default_temperature: float = _get_float("ECHO_TEMPERATURE", 0.7)
    autosave_interval_seconds: float = _get_float("ECHO_AUTOSAVE_INTERVAL", 300.0)

    def with_overrides(
        self,
        *,
        infer_url: Optional[str] = None,
        learn_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
        prefer_pipeline: Optional[bool] = None,
        default_temperature: Optional[float] = None,
        autosave_interval_seconds: Optional[float] = None,
    ) -> "EchoConfig":
        """Return a new config with provided overrides."""

        return EchoConfig(
            infer_url=infer_url or self.infer_url,
            learn_url=learn_url or self.learn_url,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            retry_backoff_seconds=
            retry_backoff_seconds if retry_backoff_seconds is not None else self.retry_backoff_seconds,
            prefer_pipeline=prefer_pipeline if prefer_pipeline is not None else self.prefer_pipeline,
            default_temperature=
            default_temperature if default_temperature is not None else self.default_temperature,
            autosave_interval_seconds=
            autosave_interval_seconds if autosave_interval_seconds is not None else self.autosave_interval_seconds,
        )


__all__ = ["EchoConfig"]
