"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state + arbitrary metadata atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata or {}},
        tmp,
    )
    tmp.replace(path)
    logger.debug("Saved checkpoint -> {}", path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (state_dict, metadata)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("metadata", {})
    return payload, {}


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the most-recently-modified usable checkpoint in a directory.

    Resolution order, newest first:
      1. ``best.pt``                (canonical name)
      2. ``run_<uuid>.pt``          (renamed after a completed training run)
      3. ``epoch_<n>.pt``           (periodic snapshots)

    Files are ranked by modification time so the most recent training
    output always wins, regardless of naming. Returns None if the
    directory has no checkpoint at all (caller must handle this — we no
    longer silently fall back to random weights).
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return None

    candidates: list[Path] = []
    for pattern in ("best.pt", "run_*.pt", "epoch_*.pt", "*.pt"):
        candidates.extend(d.glob(pattern))

    # Deduplicate while preserving discovery, then sort by mtime (newest first).
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c not in seen and c.is_file() and c.stat().st_size > 0:
            seen.add(c)
            unique.append(c)

    if not unique:
        return None

    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique[0]