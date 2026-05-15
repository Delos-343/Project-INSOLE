"""Training loop with mixed precision, early stopping, and progress callbacks."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from backend.model.architectures.classifier import MultiViewFootClassifier, build_classifier
from backend.model.config import CLASS_NAMES, ModelConfig, TrainingConfig
from backend.model.data.dataloader import build_dataloaders
from backend.model.training.losses import MultiTaskLoss
from backend.model.training.metrics import ClassificationMetrics
from backend.model.utils.checkpoint import save_checkpoint
from backend.model.utils.seeding import seed_everything


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[dict[str, Any]], None]


class Trainer:
    """Encapsulates the full training pipeline."""

    def __init__(
        self,
        train_cfg: TrainingConfig,
        model_cfg: ModelConfig | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg or ModelConfig()
        self.progress_cb = progress_cb or (lambda _: None)
        self.device = resolve_device(train_cfg.device)

        seed_everything(train_cfg.random_state)

        # --------- Data ---------
        logger.info("Building dataloaders from {}", train_cfg.data_dir)
        self.loaders = build_dataloaders(train_cfg)
        train_counts = self.loaders.train_dataset.class_counts()
        logger.info("Train class distribution: {}", train_counts)

        # --------- Model ---------
        self.model: MultiViewFootClassifier = build_classifier(self.model_cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: {} params on {}", f"{n_params:,}", self.device)

        # --------- Loss ---------
        # SAFE class-frequency weighting. Previous version did 1/count which
        # exploded to inf for classes with zero samples in the training
        # split, poisoning the cross-entropy gradient and producing absurd
        # loss values (~20+) instead of the expected ln(num_classes) ≈ 1.6.
        #
        # Empty classes get weight 0 so they contribute nothing to the loss
        # (the model can't learn what's not there). Non-empty classes are
        # normalised so the mean weight across all non-empty classes is 1.
        raw = []
        for name in CLASS_NAMES:
            c = max(0, train_counts.get(name, 0))
            raw.append(0.0 if c == 0 else 1.0 / c)
        present = [w for w in raw if w > 0]
        scale = (sum(present) / len(present)) if present else 1.0
        weights = torch.tensor(
            [w / scale if w > 0 else 0.0 for w in raw], dtype=torch.float32
        ).to(self.device)
        logger.info("Class weights: {}", {n: round(w.item(), 3) for n, w in zip(CLASS_NAMES, weights)})

        self.criterion = MultiTaskLoss(
            num_classes=self.model_cfg.num_classes,
            cls_weight=train_cfg.cls_loss_weight,
            meas_weight=train_cfg.measurement_loss_weight,
            vae_weight=train_cfg.vae_loss_weight,
            class_weights=weights,
        )

        # --------- Optim ---------
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=train_cfg.mixed_precision and self.device.type == "cuda"
        )

        # --------- State ---------
        self.best_val_acc: float = 0.0
        self.patience_counter: int = 0
        self.global_step: int = 0
        self.output_dir = Path(train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- sched
    def _build_scheduler(self) -> Any:
        s = self.train_cfg.scheduler
        if s == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.train_cfg.num_epochs)
        if s == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        return ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=3)

    # ----------------------------------------------------------------- run
    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        t0 = time.time()

        for epoch in range(1, self.train_cfg.num_epochs + 1):
            train_logs = self._train_one_epoch(epoch)
            val_logs = self._evaluate(self.loaders.val, split="val")

            # Step LR scheduler.
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_logs["accuracy"])
            else:
                self.scheduler.step()

            entry = {"epoch": epoch, **train_logs, **{f"val_{k}": v for k, v in val_logs.items()}}
            history.append(entry)
            logger.info(
                "Epoch {:>3}/{}  train_loss={:.4f}  val_acc={:.4f}  val_f1={:.4f}",
                epoch,
                self.train_cfg.num_epochs,
                train_logs["loss"],
                val_logs["accuracy"],
                val_logs["macro_f1"],
            )

            # Callback for GUI/progress bar.
            self.progress_cb(
                {
                    "type": "epoch_end",
                    "epoch": epoch,
                    "total_epochs": self.train_cfg.num_epochs,
                    "metrics": entry,
                    "elapsed_sec": time.time() - t0,
                }
            )

            # ------- checkpoint + early stopping -------
            improved = val_logs["accuracy"] > self.best_val_acc
            if improved:
                self.best_val_acc = val_logs["accuracy"]
                self.patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.output_dir / "best.pt",
                    metadata={
                        "epoch": epoch,
                        "val_accuracy": val_logs["accuracy"],
                        "model_cfg": self.model_cfg.__dict__,
                    },
                )
            else:
                self.patience_counter += 1

            if epoch % self.train_cfg.save_every == 0:
                save_checkpoint(self.model, self.output_dir / f"epoch_{epoch}.pt")

            if self.patience_counter >= self.train_cfg.early_stopping_patience:
                logger.warning("Early stopping triggered at epoch {}", epoch)
                break

        # Final eval on test set.
        test_logs = self._evaluate(self.loaders.test, split="test")
        logger.info("Final test accuracy: {:.4f}", test_logs["accuracy"])

        self.progress_cb(
            {
                "type": "done",
                "best_val_acc": self.best_val_acc,
                "test_metrics": test_logs,
                "history": history,
            }
        )

        return {
            "history": history,
            "best_val_accuracy": self.best_val_acc,
            "test_metrics": test_logs,
        }

    # ----------------------------------------------------------------- step
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        metrics = ClassificationMetrics(self.model_cfg.num_classes)

        for batch in self.loaders.train:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                self.device.type, enabled=self.scaler.is_enabled()
            ):
                outputs = self.model(
                    lateral=batch["lateral"],
                    top=batch["top"],
                    back=batch["back"],
                    measurements=batch["measurements"],
                    measurement_mask=batch["measurement_mask"],
                    labels=batch["label"],
                    return_vae=True,
                )
                loss, _ = self.criterion(outputs, batch)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            self.global_step += 1
            metrics.update(outputs["logits"].detach(), batch["label"])

            if self.global_step % self.train_cfg.log_every == 0:
                self.progress_cb(
                    {
                        "type": "step",
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": float(loss.item()),
                        "running_acc": metrics.accuracy(),
                    }
                )

        return {
            "loss": running_loss / max(1, n_batches),
            "accuracy": metrics.accuracy(),
            "macro_f1": metrics.macro_f1(),
        }

    @torch.no_grad()
    def _evaluate(self, loader, split: str) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        metrics = ClassificationMetrics(self.model_cfg.num_classes)

        for batch in loader:
            batch = self._move_batch(batch)
            outputs = self.model(
                lateral=batch["lateral"],
                top=batch["top"],
                back=batch["back"],
                measurements=batch["measurements"],
                measurement_mask=batch["measurement_mask"],
                labels=batch["label"],
                return_vae=False,
            )
            loss, _ = self.criterion(outputs, batch)
            running_loss += float(loss.item())
            n_batches += 1
            metrics.update(outputs["logits"], batch["label"])

        return {
            "loss": running_loss / max(1, n_batches),
            **metrics.to_dict(),
        }

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }


# ---------------------------------------------------------------------------
# Convenience entrypoint
# ---------------------------------------------------------------------------
def train(
    train_cfg: TrainingConfig | None = None,
    model_cfg: ModelConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run training with the given configs."""
    return Trainer(train_cfg or TrainingConfig(), model_cfg, progress_cb).fit()
