"""
Inference engine — SHEET-LOOKUP-FIRST (revision 2026-05).

Order of authority:
  1. Patient code resolves in the consolidated sheet
        -> deterministic dual-rule classification (arch authoritative,
           heel angle corroborating). Source = "sheet". 100% confidence.
  2. Patient not in sheet (or no code)
        -> model estimates arch height from images, classified by the
           same arch-height bands. Source = "image_estimated". Flagged
           assistive, honest sub-100% confidence. (Diagnostics show this
           is unreliable for this dataset — it exists only so the app
           still returns *something* for an unknown patient.)

Manual measurement entry is removed entirely (revision A + B3). The five
measurements always come from the sheet; they are never typed.
"""

from __future__ import annotations

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
from backend.model.config import ARCH_HEIGHT_BANDS, CLASS_NAMES, InferenceConfig, ModelConfig
from backend.model.data.measurement_lookup import (
    CLASS_ORDER,
    classify_by_arch_height_cm,
    lookup_patient,
)
from backend.model.data.transforms import build_eval_transform
from backend.model.training.trainer import resolve_device
from backend.model.utils.checkpoint import find_latest_checkpoint, load_checkpoint


class NoTrainedModelError(RuntimeError):
    pass


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict[str, float]
    measurements_used: dict[str, float]
    classification_source: str            # "sheet" | "image_estimated"
    arch_class: str | None
    heel_class: str | None
    rules_agree: bool
    insole_configuration: dict[str, float]
    severity_band: str
    notes: list[str]
    checkpoint_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity_band(cls: str) -> str:
    return {
        "Normal Foot": "normal",
        "Flat Arch": "moderate",
        "High Arch": "moderate",
        "Severe Flat Arch": "severe",
        "Severe High Arch": "severe",
    }.get(cls, "unknown")


class Predictor:
    def __init__(self, cfg: InferenceConfig | None = None) -> None:
        self.cfg = cfg or InferenceConfig()
        self.device = resolve_device(self.cfg.device)
        self.transform = build_eval_transform(self.cfg.image_size)
        self.checkpoint_path: Path | None = None

        model_cfg, weights = self._load_weights()
        self.model_cfg = model_cfg
        self.model = build_classifier(model_cfg).to(self.device)
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if (len(weights) - len(unexpected)) == 0:
            raise NoTrainedModelError(
                "Checkpoint loaded but zero weights bound — architecture mismatch."
            )
        self.model.eval()
        logger.info("Predictor ready (sheet-lookup-first) from {}", self.checkpoint_path)

    def _load_weights(self):
        explicit = Path(self.cfg.checkpoint_path)
        chosen = explicit if explicit.exists() else find_latest_checkpoint(explicit.parent)
        if chosen is None:
            raise NoTrainedModelError(
                f"No trained checkpoint in {explicit.parent.resolve()}."
            )
        self.checkpoint_path = chosen
        state, meta = load_checkpoint(chosen, map_location=self.device)
        mc = ModelConfig(**meta["model_cfg"]) if meta.get("model_cfg") else ModelConfig()
        return mc, state

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def predict(
        self,
        lateral_path: str | Path | None,
        top_path: str | Path | None,
        back_path: str | Path | None,
        patient_code: str | None = None,
        sheet_path: str | Path | None = None,
        **_ignored: Any,                       # tolerate legacy 'measurements' kw
    ) -> PredictionResult:
        notes: list[str] = []

        # ---- 1. Authoritative path: sheet lookup by patient code ----
        lk = lookup_patient(patient_code, sheet_path=sheet_path)
        if lk.found:
            headline = lk.headline_class
            idx = CLASS_NAMES.index(headline)
            notes.append(lk.note)
            if not lk.rules_agree:
                notes.append(
                    "Two independent measurement rules disagree by one "
                    "class; flagged for clinician review."
                )
            # Insole config still comes from the generative head (uses images
            # if present; zeros otherwise — it is advisory, not the class).
            insole = self._insole_config(lateral_path, top_path, back_path,
                                         lk.measurements)
            probs = {c: (1.0 if c == headline else 0.0) for c in CLASS_NAMES}
            return PredictionResult(
                predicted_class=headline,
                predicted_class_idx=idx,
                confidence=1.0,
                class_probabilities=probs,
                measurements_used=lk.measurements,
                classification_source="sheet",
                arch_class=lk.arch_class,
                heel_class=lk.heel_class,
                rules_agree=lk.rules_agree,
                insole_configuration=insole,
                severity_band=_severity_band(headline),
                notes=notes,
                checkpoint_used=str(self.checkpoint_path),
            )

        # ---- 2. Fallback: image estimate (assistive, flagged) ----
        notes.append(lk.note)
        notes.append(
            "Patient not found in the consolidated sheet, so arch height "
            "was ESTIMATED from images. This is assistive only and "
            "unreliable for this dataset — confirm with the sheet/clinic."
        )
        lat, lp = self._img(lateral_path)
        top, tp = self._img(top_path)
        bak, bp = self._img(back_path)
        out = self.model(
            lateral=lat.unsqueeze(0).to(self.device),
            top=top.unsqueeze(0).to(self.device),
            back=bak.unsqueeze(0).to(self.device),
            measurements=torch.zeros(1, 5, device=self.device),
            measurement_mask=torch.tensor([[0.0]], device=self.device),
        )
        est = out["measurements_hat"].squeeze(0).cpu().numpy()
        est_map = {k: float(est[i]) for i, k in enumerate(MeasurementHead.MEASUREMENT_NAMES)}
        arch_cm = float(est_map.get("arch_height_cm", 0.0))
        cls = classify_by_arch_height_cm(arch_cm)
        idx = CLASS_NAMES.index(cls)
        conf, probs = self._soft_conf(arch_cm, idx)
        insole = {
            n: float(out["insole_config"].squeeze(0).cpu().numpy()[i])
            for i, n in enumerate(InsoleConfigHead.OUTPUT_NAMES)
        }
        return PredictionResult(
            predicted_class=cls,
            predicted_class_idx=idx,
            confidence=conf,
            class_probabilities=probs,
            measurements_used=est_map,
            classification_source="image_estimated",
            arch_class=cls,
            heel_class=None,
            rules_agree=True,
            insole_configuration=insole,
            severity_band=_severity_band(cls),
            notes=notes,
            checkpoint_used=str(self.checkpoint_path),
        )

    # ----------------------------------------------------------- helpers
    def _insole_config(self, lp, tp, bp, meas: dict) -> dict[str, float]:
        lat, _ = self._img(lp)
        top, _ = self._img(tp)
        bak, _ = self._img(bp)
        vec = torch.zeros(1, 5, device=self.device)
        order = MeasurementHead.MEASUREMENT_NAMES
        for i, k in enumerate(order):
            if k in meas:
                vec[0, i] = float(meas[k])
        out = self.model(
            lateral=lat.unsqueeze(0).to(self.device),
            top=top.unsqueeze(0).to(self.device),
            back=bak.unsqueeze(0).to(self.device),
            measurements=vec,
            measurement_mask=torch.tensor([[1.0]], device=self.device),
        )
        ic = out["insole_config"].squeeze(0).cpu().numpy()
        return {n: float(ic[i]) for i, n in enumerate(InsoleConfigHead.OUTPUT_NAMES)}

    def _img(self, path):
        if path is None or not Path(path).exists():
            return self.transform(
                Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), (0, 0, 0))
            ), False
        return self.transform(Image.open(path).convert("RGB")), True

    def _soft_conf(self, arch_cm: float, idx: int):
        import math
        sigma = 0.6
        def cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        mass = {}
        for name in CLASS_NAMES:
            lo, hi = ARCH_HEIGHT_BANDS[name]
            mass[name] = max(0.0, cdf((hi - arch_cm) / sigma) - cdf((lo - arch_cm) / sigma))
        tot = sum(mass.values()) or 1.0
        probs = {k: v / tot for k, v in mass.items()}
        return float(probs[CLASS_NAMES[idx]]), probs


# ---------------------------------------------------------------------------
# Module-level convenience function.
#
# The package __init__ files (backend/model/__init__.py and
# backend/model/inference/__init__.py) import `predict` by name. The
# revision rewrite must keep exporting it or the whole backend fails to
# import at startup. Signature updated for the sheet-lookup-first design:
# patient_code is the primary key; `measurements` is accepted and ignored
# for backward compatibility with any old caller.
# ---------------------------------------------------------------------------
def predict(
    lateral_path: str | Path | None = None,
    top_path: str | Path | None = None,
    back_path: str | Path | None = None,
    patient_code: str | None = None,
    sheet_path: str | Path | None = None,
    cfg: InferenceConfig | None = None,
    **_legacy: Any,
) -> PredictionResult:
    """One-shot predict helper. Builds a Predictor and runs it once."""
    return Predictor(cfg).predict(
        lateral_path=lateral_path,
        top_path=top_path,
        back_path=back_path,
        patient_code=patient_code,
        sheet_path=sheet_path,
    )
