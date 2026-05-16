"""
Training endpoints — start a run, query status (with live progress), list history.

Training is launched in a background thread so the HTTP request returns
immediately. A progress callback writes each epoch's metrics into the
TrainingRun.history column so the GUI can poll for live updates.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from backend.database.connection import db_session, get_db
from backend.database.models import TrainingRun
from backend.database.repositories.training_run_repo import TrainingRunRepository
from backend.database.schemas import TrainingRequest, TrainingStatusOut
from backend.model.config import ModelConfig, TrainingConfig

router = APIRouter()
_active_runs: dict[str, threading.Thread] = {}


def _config_to_jsonable(cfg: Any) -> dict:
    out: dict = {}
    for k, v in cfg.__dict__.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _run_training_thread(run_id: str, training_cfg: TrainingConfig, model_cfg: ModelConfig) -> None:
    from backend.model.training.trainer import Trainer

    with db_session() as db:
        TrainingRunRepository(db).mark_running(run_id)

    def progress_cb(payload: dict) -> None:
        if payload.get("type") != "epoch_end":
            return
        try:
            with db_session() as db:
                row = db.get(TrainingRun, run_id)
                if row is None:
                    return
                history = list(row.history or [])
                history.append(payload["metrics"])
                row.history = history
                row.total_epochs = len(history)
                val_acc = payload["metrics"].get("val_accuracy")
                if val_acc is not None and (
                    row.best_val_accuracy is None or val_acc > row.best_val_accuracy
                ):
                    row.best_val_accuracy = float(val_acc)
                db.commit()
        except Exception as exc:
            logger.warning("Progress callback DB write failed: {}", exc)

    try:
        trainer = Trainer(training_cfg, model_cfg, progress_cb=progress_cb)
        result = trainer.fit()

        out_dir = Path(training_cfg.output_dir)
        best_src = out_dir / "best.pt"

        # ── CRITICAL FIX ──────────────────────────────────────────────
        # Previously we did best_src.replace(best_dst), which DELETED
        # best.pt by renaming it to run_<uuid>.pt. The Predictor then
        # looked for best.pt, didn't find it, and silently ran on random
        # weights. Now we COPY (not move): best.pt stays as the canonical
        # checkpoint the Predictor always loads, and we additionally keep
        # a run-tagged archival copy.
        # ──────────────────────────────────────────────────────────────
        checkpoint_path = str(best_src)
        if best_src.exists():
            run_archive = out_dir / f"run_{run_id}.pt"
            try:
                shutil.copy2(best_src, run_archive)
            except Exception as exc:
                logger.warning("Could not archive run checkpoint: {}", exc)

        with db_session() as db:
            TrainingRunRepository(db).mark_completed(
                run_id,
                best_val_accuracy=result["best_val_accuracy"],
                test_accuracy=result["test_metrics"].get("accuracy"),
                macro_f1=result["test_metrics"].get("macro_f1"),
                total_epochs=len(result["history"]),
                history=result["history"],
                checkpoint_path=checkpoint_path,
                model_version=f"v0.1.0+{run_id[:8]}",
            )
        logger.info("Training run {} completed. best.pt retained at {}", run_id, best_src)

    except Exception:
        logger.exception("Training run {} failed", run_id)
        with db_session() as db:
            TrainingRunRepository(db).mark_failed(run_id)
    finally:
        _active_runs.pop(run_id, None)


@router.post("/runs", response_model=TrainingStatusOut, status_code=202)
async def start_training(
    req: TrainingRequest,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingStatusOut:
    training_cfg = TrainingConfig(
        data_dir=Path(req.data_dir) if req.data_dir else Path("data"),
        batch_size=req.batch_size,
        num_epochs=req.num_epochs,
        learning_rate=req.learning_rate,
        image_size=req.image_size,
        use_augmentation=req.use_augmentation,
        train_split=req.train_split,
        val_split=req.val_split,
    )
    model_cfg = ModelConfig(use_generative_branch=req.use_generative_branch)

    repo = TrainingRunRepository(db)
    row = repo.create(
        name=req.name,
        training_config=_config_to_jsonable(training_cfg),
        model_config=_config_to_jsonable(model_cfg),
    )

    t = threading.Thread(
        target=_run_training_thread, args=(row.id, training_cfg, model_cfg), daemon=True
    )
    _active_runs[row.id] = t
    t.start()
    return _row_to_status(row)


@router.get("/runs", response_model=list[TrainingStatusOut])
async def list_runs(limit: int = 20, db: Session = Depends(get_db)) -> Sequence[TrainingStatusOut]:
    rows = TrainingRunRepository(db).list_recent(limit)
    return [_row_to_status(r) for r in rows]


@router.get("/runs/{run_id}", response_model=TrainingStatusOut)
async def get_run(run_id: str, db: Session = Depends(get_db)) -> TrainingStatusOut:
    row = TrainingRunRepository(db).get(run_id)
    if row is None:
        raise HTTPException(404, detail="Training run not found.")
    return _row_to_status(row)


def _row_to_status(row) -> TrainingStatusOut:
    return TrainingStatusOut(
        id=row.id,
        name=row.name,
        status=row.status,
        best_val_accuracy=row.best_val_accuracy,
        test_accuracy=row.test_accuracy,
        macro_f1=row.macro_f1,
        total_epochs=row.total_epochs,
        trained_minutes=row.trained_minutes,
        started_at=row.started_at,
        finished_at=row.finished_at,
        checkpoint_path=row.checkpoint_path,
        history=row.history,
    )