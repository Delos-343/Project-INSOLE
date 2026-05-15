"""
Training tab — the SECONDARY tab.

Lets the user point at the data folder, set hyper-parameters, start training,
and watch a live progress bar + console log.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_CONFIG
from app.ui.theme.colors import PALETTE as P
from app.ui.widgets.log_console import LogConsole
from app.ui.workers.training_worker import TrainingWorker


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainingWorker | None = None
        self._build()

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("trainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(16)
        splitter.setStyleSheet(
            f"""
            QSplitter#trainSplitter {{
                background-color: transparent;
            }}
            QSplitter#trainSplitter::handle {{
                background-color: transparent;
                margin: 0 6px;
            }}
            QSplitter#trainSplitter::handle:hover {{
                background-color: {P.border};
                border-radius: 2px;
            }}
            """
        )

        # ---------- LEFT: scrollable config area ----------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(2, 2, 8, 2)
        left_l.setSpacing(14)

        # Header
        h_sub = QLabel("Train")
        h_sub.setObjectName("subtitleLabel")
        h_title = QLabel("Model training pipeline")
        h_title.setObjectName("titleLabel")
        h_hint = QLabel(
            "Train the multi-view foot classifier on the dataset under "
            "<i>data/</i>. The pipeline expects subfolders "
            "<i>Heel/, Flat/, Normal/, Sheet/</i> per the project brief."
        )
        h_hint.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
        h_hint.setWordWrap(True)
        h_hint.setTextFormat(Qt.TextFormat.RichText)
        left_l.addWidget(h_sub)
        left_l.addWidget(h_title)
        left_l.addWidget(h_hint)

        # ---- Data ----
        data_box = QGroupBox("Data source")
        data_l = QVBoxLayout(data_box)
        data_l.setContentsMargins(14, 24, 14, 14)
        data_l.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.data_dir_edit = QLineEdit(str(APP_CONFIG.default_data_dir.resolve()))
        row.addWidget(self.data_dir_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        data_l.addLayout(row)

        self.scan_label = QLabel("Click 'Scan dataset' to inspect contents.")
        self.scan_label.setStyleSheet(f"color: {P.text_secondary}; font-size: 11px;")
        self.scan_label.setWordWrap(True)
        data_l.addWidget(self.scan_label)

        scan_btn = QPushButton("Scan dataset")
        scan_btn.clicked.connect(self._on_scan)
        data_l.addWidget(scan_btn)
        left_l.addWidget(data_box)

        # ---- Hyper-parameters ----
        hyper_box = QGroupBox("Hyper-parameters")
        hf = QFormLayout(hyper_box)
        hf.setContentsMargins(14, 24, 14, 14)
        hf.setSpacing(8)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 256)
        self.batch_size.setValue(16)
        hf.addRow("Batch size", self.batch_size)

        self.num_epochs = QSpinBox()
        self.num_epochs.setRange(1, 1000)
        self.num_epochs.setValue(50)
        hf.addRow("Epochs", self.num_epochs)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-6, 1.0)
        self.lr.setDecimals(6)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(1e-4)
        hf.addRow("Learning rate", self.lr)

        self.image_size = QSpinBox()
        self.image_size.setRange(64, 1024)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(256)
        hf.addRow("Image size", self.image_size)

        self.use_aug = QCheckBox("Data augmentation")
        self.use_aug.setChecked(True)
        hf.addRow("", self.use_aug)

        self.use_gen = QCheckBox("Generative VAE branch")
        self.use_gen.setChecked(True)
        hf.addRow("", self.use_gen)

        left_l.addWidget(hyper_box)

        # ---- CTA ----
        cta_row = QHBoxLayout()
        cta_row.setSpacing(8)
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self._on_start)
        cta_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        cta_row.addWidget(self.stop_btn)
        cta_row.addStretch()
        left_l.addLayout(cta_row)

        # Progress
        prog_box = QFrame()
        prog_l = QVBoxLayout(prog_box)
        prog_l.setContentsMargins(0, 0, 0, 0)
        prog_l.setSpacing(6)
        self.epoch_label = QLabel("No training in progress.")
        self.epoch_label.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog_l.addWidget(self.epoch_label)
        prog_l.addWidget(self.progress)
        left_l.addWidget(prog_box)

        # Metric chips
        chips = QHBoxLayout()
        chips.setSpacing(10)
        self.acc_chip = self._metric_chip("VAL ACCURACY", "—")
        self.f1_chip = self._metric_chip("MACRO F1", "—")
        self.loss_chip = self._metric_chip("LOSS", "—")
        chips.addWidget(self.acc_chip)
        chips.addWidget(self.f1_chip)
        chips.addWidget(self.loss_chip)
        left_l.addLayout(chips)

        left_l.addStretch()
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ---------- RIGHT: console ----------
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(8)
        log_label = QLabel("Training log")
        log_label.setObjectName("sectionLabel")
        right_l.addWidget(log_label)
        self.console = LogConsole()
        right_l.addWidget(self.console, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([600, 800])
        outer.addWidget(splitter, 1)

    def _metric_chip(self, label: str, value: str) -> QFrame:
        f = QFrame()
        f.setStyleSheet(
            f"QFrame {{ background-color: {P.bg_secondary}; "
            f"border: 1px solid {P.border}; border-radius: 10px; padding: 10px; }}"
        )
        l = QVBoxLayout(f)
        l.setContentsMargins(12, 10, 12, 10)
        l.setSpacing(2)
        title = QLabel(label)
        title.setStyleSheet(
            f"color: {P.text_muted}; font-size: 9px; letter-spacing: 2px; font-weight: 600;"
        )
        val = QLabel(value)
        val.setStyleSheet(f"color: {P.text_primary}; font-size: 18px; font-weight: 700;")
        val.setObjectName("metricValue")
        l.addWidget(title)
        l.addWidget(val)
        f._value_label = val  # type: ignore[attr-defined]
        return f

    # -------------------------------------------------------------- events
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select data folder", self.data_dir_edit.text())
        if path:
            self.data_dir_edit.setText(path)

    def _on_scan(self) -> None:
        from backend.model.data.dataset import IMAGE_EXTS, FootClassificationDataset

        data_dir = Path(self.data_dir_edit.text())
        if not data_dir.exists():
            self.scan_label.setText(f"<span style='color: {P.danger}'>Folder does not exist.</span>")
            return
        n_imgs = sum(1 for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        try:
            ds = FootClassificationDataset(data_dir=data_dir, transform=None)
            counts = ds.class_counts()
            distrib = ", ".join(f"{k}: {v}" for k, v in counts.items())
            self.scan_label.setText(
                f"<b>{len(ds)}</b> patient samples discovered "
                f"({n_imgs} raw images total).<br>{distrib}"
            )
        except Exception as exc:
            self.scan_label.setText(f"<span style='color: {P.danger}'>Scan failed: {exc}</span>")

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        data_dir = Path(self.data_dir_edit.text())
        if not data_dir.exists():
            QMessageBox.warning(self, "Cannot start", f"Data folder does not exist: {data_dir}")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.epoch_label.setText("Starting…")
        self.console.clear()

        self._worker = TrainingWorker(
            data_dir=data_dir,
            batch_size=self.batch_size.value(),
            num_epochs=self.num_epochs.value(),
            learning_rate=self.lr.value(),
            image_size=self.image_size.value(),
            use_augmentation=self.use_aug.isChecked(),
            use_generative_branch=self.use_gen.isChecked(),
            parent=self,
        )
        self._worker.log.connect(self.console.append_line)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_thread_finished)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Stop requested",
                "Training will stop after the current epoch completes.",
            )
            self.stop_btn.setEnabled(False)

    def _on_epoch(self, payload: dict) -> None:
        epoch = payload["epoch"]
        total = payload["total_epochs"]
        m = payload["metrics"]
        self.progress.setRange(0, total)
        self.progress.setValue(epoch)
        self.epoch_label.setText(
            f"Epoch {epoch}/{total} — val_acc {m['val_accuracy']:.3f} • "
            f"f1 {m['val_macro_f1']:.3f} • elapsed {payload['elapsed_sec']:.0f}s"
        )
        self.acc_chip._value_label.setText(f"{m['val_accuracy']*100:.1f}%")    # type: ignore[attr-defined]
        self.f1_chip._value_label.setText(f"{m['val_macro_f1']*100:.1f}%")     # type: ignore[attr-defined]
        self.loss_chip._value_label.setText(f"{m['loss']:.4f}")                # type: ignore[attr-defined]

    def _on_done(self, result: dict) -> None:
        QMessageBox.information(
            self,
            "Training complete",
            f"Best val accuracy: {result['best_val_accuracy']*100:.2f}%\n"
            f"Test accuracy:     {result['test_metrics']['accuracy']*100:.2f}%",
        )

    def _on_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Training failed", msg)

    def _on_thread_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
