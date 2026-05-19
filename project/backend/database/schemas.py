"""Pydantic schemas used by the FastAPI layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.database.models import FootClass, SeverityBand, TrainStatus


class MeasurementsIn(BaseModel):
    calcaneal_inclination_deg: float | None = None
    heel_angle_deg:            float | None = None
    arch_height_cm:            float | None = None
    kite_angle_deg:            float | None = None
    first_metatarsal_talus_deg: float | None = None

    def is_any_provided(self) -> bool:
        return any(v is not None for v in self.model_dump().values())


class InsoleConfigOut(BaseModel):
    arch_support_height:    float
    heel_cup_depth:         float
    medial_post_strength:   float
    lateral_wedge_strength: float
    forefoot_cushioning:    float


class ClassificationOut(BaseModel):
    """Response payload returned by /api/classify."""

    id: str | None = None
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict[str, float]
    severity_band: str
    rule_based_label: str

    measurements_predicted: dict[str, float]            # == measurements_used
    measurements_provided:  dict[str, float | None]

    insole_configuration: InsoleConfigOut
    notes: list[str] = Field(default_factory=list)
    inference_time_ms: int | None = None
    model_version: str | None = None

    # --- Measurement-first additions ---
    # 'measured'         -> authoritative, deterministic rule on clinician input
    # 'image_estimated'  -> assistive, model-estimated arch height, lower trust
    classification_source: str = "measured"
    measurements_estimated: dict[str, float] = Field(default_factory=dict)

    # --- Sheet-lookup revision: dual-rule detail ---
    # arch_class  -> authoritative class from the arch-height bands
    # heel_class  -> corroborating class from the heel-angle bands
    # rules_agree -> False flags a one-class boundary disagreement
    arch_class: str | None = None
    heel_class: str | None = None
    rules_agree: bool = True


class PatientIn(BaseModel):
    code: str
    age: int | None = None
    sex: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    notes: str | None = None


class PatientOut(PatientIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime


class TrainingRequest(BaseModel):
    name: str | None = None
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    image_size: int = 256
    use_augmentation: bool = True
    use_generative_branch: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    data_dir: str | None = None


class TrainingStatusOut(BaseModel):
    id: str
    name: str | None
    status: TrainStatus
    best_val_accuracy: float | None
    test_accuracy: float | None
    macro_f1: float | None
    total_epochs: int | None
    trained_minutes: int | None
    started_at: datetime | None
    finished_at: datetime | None
    checkpoint_path: str | None
    history: list[dict] | None = None


class HealthOut(BaseModel):
    status: str = "ok"
    service: str = "insole-foot-classification"
    version: str = "0.2.0"
    model_loaded: bool
    device: str
    db_connected: bool
