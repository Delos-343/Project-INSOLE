"""
POST /api/classify — sheet-lookup-first inference (revision 2026-05).

The route no longer accepts manually-entered measurements. Classification
is driven by patient_code -> consolidated-sheet lookup on the backend.
Images are still accepted (used for the insole-config head and for the
estimate fallback when a patient is absent from the sheet).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.repositories.classification_repo import ClassificationRepository
from backend.database.repositories.patient_repo import PatientRepository
from backend.database.schemas import ClassificationOut, InsoleConfigOut
from backend.server.dependencies import get_predictor
from backend.server.utils.file_handler import save_upload

router = APIRouter()


@router.post("/classify", response_model=ClassificationOut)
async def classify(
    lateral: UploadFile | None = File(None),
    top: UploadFile | None = File(None),
    back: UploadFile | None = File(None),
    patient_code: str | None = Form(None),
    predictor=Depends(get_predictor),
    db: Session = Depends(get_db),
) -> ClassificationOut:
    if not patient_code and lateral is None and top is None and back is None:
        raise HTTPException(
            400,
            detail="Provide a patient code (preferred) or at least one image.",
        )

    lat_path = await save_upload(lateral, "lat") if lateral else None
    top_path = await save_upload(top, "top") if top else None
    bak_path = await save_upload(back, "back") if back else None

    t0 = time.perf_counter()
    try:
        result = predictor.predict(
            lateral_path=lat_path,
            top_path=top_path,
            back_path=bak_path,
            patient_code=patient_code,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, detail=f"Inference failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    pid = None
    if patient_code:
        try:
            pid = PatientRepository(db).upsert_by_code(patient_code).id
        except Exception as exc:
            logger.warning("Patient upsert failed: {}", exc)

    used = result.measurements_used
    try:
        row = ClassificationRepository(db).create(
            patient_id=pid,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            class_probs=result.class_probabilities,
            severity_band=result.severity_band,
            rule_based_label=result.arch_class or result.predicted_class,
            notes="\n".join(result.notes) if result.notes else None,
            lateral_image_path=str(lat_path) if lat_path else None,
            top_image_path=str(top_path) if top_path else None,
            back_image_path=str(bak_path) if bak_path else None,
            calcaneal_inclination_deg=used.get("calcaneal_inclination_deg"),
            heel_angle_deg=used.get("heel_angle_deg"),
            arch_height_cm=used.get("arch_height_cm"),
            kite_angle_deg=used.get("kite_angle_deg"),
            first_metatarsal_talus_deg=used.get("first_metatarsal_talus_deg"),
            measurements_were_provided=(result.classification_source == "sheet"),
            arch_support_height=result.insole_configuration.get("arch_support_height"),
            heel_cup_depth=result.insole_configuration.get("heel_cup_depth"),
            medial_post_strength=result.insole_configuration.get("medial_post_strength"),
            lateral_wedge_strength=result.insole_configuration.get("lateral_wedge_strength"),
            forefoot_cushioning=result.insole_configuration.get("forefoot_cushioning"),
            inference_time_ms=elapsed_ms,
            model_version="v0.3.0-sheet-lookup",
        )
        record_id = row.id
    except Exception as exc:
        logger.warning("Could not persist classification: {}", exc)
        record_id = None

    return ClassificationOut(
        id=record_id,
        predicted_class=result.predicted_class,
        predicted_class_idx=result.predicted_class_idx,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        severity_band=result.severity_band,
        rule_based_label=result.arch_class or result.predicted_class,
        measurements_predicted=used,
        measurements_provided={},
        insole_configuration=InsoleConfigOut(**result.insole_configuration),
        notes=result.notes,
        inference_time_ms=elapsed_ms,
        model_version="v0.3.0-sheet-lookup",
        classification_source=result.classification_source,
        measurements_estimated=(
            used if result.classification_source == "image_estimated" else {}
        ),
        arch_class=result.arch_class,
        heel_class=result.heel_class,
        rules_agree=result.rules_agree,
    )
