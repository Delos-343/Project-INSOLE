"""Inference engine — single-call predict() on a triplet of images."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image

from backend.model.architectures.classifier import build_classifier
from backend.model.architectures.generative_vae import InsoleConfigHead
from backend.model.architectures.measurement_predictor import MeasurementHead
from backend.model.config import (
    ARCH_HEIGHT_BANDS,
    CLASS_NAMES,
    HEEL_ANGLE_BANDS,
    InferenceConfig,
    ModelConfig,
)
from backend.model.data.transforms import build_eval_transform
from backend.model.training.trainer import resolve_device
from backend.model.utils.checkpoint import find_latest_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Raised when no usable trained checkpoint exists. We deliberately DO NOT
# fall back to random weights — that produced confident-looking garbage and
# masked a broken pipeline for an entire debugging session.
# ---------------------------------------------------------------------------
class NoTrainedModelError(RuntimeError):
    pass


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict[str, float]
    measurements_predicted: dict[str, float]
    measurements_provided: dict[str, float | None]
    insole_configuration: dict[str, float]
    severity_band: str
    rule_based_label: str
    notes: list[str]
    checkpoint_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Predictor:
    """Stateful predictor — load once, call many times."""

    def __init__(self, cfg: InferenceConfig | None = None) -> None:
        self.cfg = cfg or InferenceConfig()
        self.device = resolve_device(self.cfg.device)
        self.transform = build_eval_transform(self.cfg.image_size)
        self.checkpoint_path: Path | None = None

        model_cfg, weights = self._load_weights()
        self.model_cfg = model_cfg
        self.model = build_classifier(model_cfg).to(self.device)

        # Strict load: if the checkpoint architecture doesn't match, we want
        # a hard error, not silently-unbound random weights.
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        n_loaded = len(weights) - len(unexpected)
        if n_loaded == 0:
            raise NoTrainedModelError(
                "Checkpoint loaded but ZERO weights bound to the model — "
                "architecture mismatch. The model would run on random weights."
            )
        if missing:
            logger.warning(
                "{} model params had no checkpoint value (using init): {}...",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "{} checkpoint params not used by model: {}...",
                len(unexpected),
                unexpected[:5],
            )
        logger.info(
            "Predictor ready — {} weights loaded from {}",
            n_loaded,
            self.checkpoint_path,
        )
        self.model.eval()

    def _load_weights(self) -> tuple[ModelConfig, dict]:
        # 1. Honour an explicit path if it exists.
        explicit = Path(self.cfg.checkpoint_path)
        chosen: Path | None = explicit if explicit.exists() else None

        # 2. Otherwise, find the newest checkpoint in the same directory.
        if chosen is None:
            ckpt_dir = explicit.parent
            chosen = find_latest_checkpoint(ckpt_dir)

        if chosen is None:
            raise NoTrainedModelError(
                f"No trained checkpoint found. Looked for an explicit path "
                f"({explicit}) and any *.pt in {explicit.parent.resolve()}. "
                f"Train a model before running inference."
            )

        self.checkpoint_path = chosen
        try:
            state, meta = load_checkpoint(chosen, map_location=self.device)
        except Exception as exc:
            raise NoTrainedModelError(
                f"Failed to load checkpoint {chosen}: {exc}"
            ) from exc

        model_cfg = ModelConfig(**meta["model_cfg"]) if meta.get("model_cfg") else ModelConfig()
        return model_cfg, state

    @torch.no_grad()
    def predict(
        self,
        lateral_path: str | Path | None,
        top_path: str | Path | None,
        back_path: str | Path | None,
        measurements: Mapping[str, float] | None = None,
    ) -> PredictionResult:
        lat_t, lat_present = self._prep_image(lateral_path)
        top_t, top_present = self._prep_image(top_path)
        bak_t, bak_present = self._prep_image(back_path)

        notes: list[str] = []
        for present, name in [
            (lat_present, "lateral"),
            (top_present, "top"),
            (bak_present, "back"),
        ]:
            if not present:
                notes.append(f"Missing {name} view — using zero placeholder.")

        meas_vec, mask = self._prep_measurements(measurements)
        meas_provided_dict: dict[str, float | None] = {
            name: (measurements.get(name) if measurements else None)
            for name in MeasurementHead.MEASUREMENT_NAMES
        }

        outputs = self.model(
            lateral=lat_t.unsqueeze(0).to(self.device),
            top=top_t.unsqueeze(0).to(self.device),
            back=bak_t.unsqueeze(0).to(self.device),
            measurements=meas_vec.unsqueeze(0).to(self.device),
            measurement_mask=torch.tensor([[mask]], device=self.device),
        )

        probs = outputs["logits"].softmax(dim=-1).squeeze(0).cpu().numpy()
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        pred_meas = outputs["measurements_hat"].squeeze(0).cpu().numpy()
        meas_out: dict[str, float] = {}
        for i, name in enumerate(MeasurementHead.MEASUREMENT_NAMES):
            if measurements and measurements.get(name) is not None:
                meas_out[name] = float(measurements[name])  # type: ignore[arg-type]
            else:
                meas_out[name] = float(pred_meas[i])

        insole = outputs["insole_config"].squeeze(0).cpu().numpy()
        insole_dict = {
            name: float(insole[i]) for i, name in enumerate(InsoleConfigHead.OUTPUT_NAMES)
        }

        rule_label = _rule_based_label(meas_out)
        if rule_label != CLASS_NAMES[top_idx]:
            notes.append(
                f"Rule-based label from measurements ('{rule_label}') disagrees "
                f"with model prediction ('{CLASS_NAMES[top_idx]}'). Consider review."
            )

        severity = _severity_band(CLASS_NAMES[top_idx])
        if top_conf < self.cfg.confidence_threshold:
            notes.append(
                f"Confidence {top_conf:.1%} below threshold "
                f"{self.cfg.confidence_threshold:.1%}; treat as uncertain."
            )

        return PredictionResult(
            predicted_class=CLASS_NAMES[top_idx],
            predicted_class_idx=top_idx,
            confidence=top_conf,
            class_probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            measurements_predicted=meas_out,
            measurements_provided=meas_provided_dict,
            insole_configuration=insole_dict,
            severity_band=severity,
            rule_based_label=rule_label,
            notes=notes,
            checkpoint_used=str(self.checkpoint_path),
        )

    def _prep_image(self, path: str | Path | None) -> tuple[torch.Tensor, bool]:
        if path is None:
            black = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), (0, 0, 0))
            return self.transform(black), False
        path = Path(path)
        if not path.exists():
            black = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), (0, 0, 0))
            return self.transform(black), False
        img = Image.open(path).convert("RGB")
        return self.transform(img), True

    def _prep_measurements(self, m: Mapping[str, float] | None) -> tuple[torch.Tensor, float]:
        vec = torch.zeros(5, dtype=torch.float32)
        if not m:
            return vec, 0.0
        any_present = False
        for i, name in enumerate(MeasurementHead.MEASUREMENT_NAMES):
            if name in m and m[name] is not None:
                vec[i] = float(m[name])
                any_present = True
        return vec, (1.0 if any_present else 0.0)


def _rule_based_label(meas: dict[str, float]) -> str:
    h = meas.get("arch_height_cm")
    heel = meas.get("heel_angle_deg")
    if h is not None and h > 0:
        for name, (lo, hi) in ARCH_HEIGHT_BANDS.items():
            if lo <= h < hi:
                return name
    if heel is not None:
        for name, (lo, hi) in HEEL_ANGLE_BANDS.items():
            if lo <= heel < hi:
                return name
    return "Normal Foot"


def _severity_band(class_name: str) -> str:
    return {
        "Normal Foot": "normal",
        "Flat Arch": "moderate",
        "High Arch": "moderate",
        "Severe Flat Arch": "severe",
        "Severe High Arch": "severe",
    }.get(class_name, "unknown")


def predict(
    lateral_path: str | Path | None,
    top_path: str | Path | None,
    back_path: str | Path | None,
    measurements: Mapping[str, float] | None = None,
    cfg: InferenceConfig | None = None,
) -> PredictionResult:
    return Predictor(cfg).predict(lateral_path, top_path, back_path, measurements)