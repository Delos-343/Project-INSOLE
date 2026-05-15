"""
Classification tab — the PRIMARY tab.

Layout mirrors the "EXPECTED OUTPUT (in exe file)" diagram from the brief:

    +-------------------+-------------------+-------------------+
    |   INSERT IMAGE    |   INSERT IMAGE    |   INSERT IMAGE    |
    |                   |                   |                   |
    |  Lateral view     |   Top view        |   Back view       |
    +-------------------+-------------------+-------------------+
    |  [ Measurements ]                                         |
    |  [ Patient code ] [ ANALYZE FOOT ]                        |
    +-----------------------------------------------------------+

With the Results panel on the right side (split view).

The left pane is wrapped in a QScrollArea so the CTA stays reachable
even on short displays. A visible gap separates the left and right panes
via a styled QSplitter handle.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_CONFIG
from app.ui.theme.colors import PALETTE as P
from app.ui.widgets.image_dropzone import ImageDropZone
from app.ui.widgets.measurement_panel import MeasurementPanel
from app.ui.widgets.results_panel import ResultsPanel
from app.ui.workers.inference_worker import InferenceWorker


class ClassificationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._measurements: dict = {}
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("classifySplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(16)             # ← visible gap between panes
        splitter.setStyleSheet(
            # The handle is the visible gap; the panes themselves are flush against it.
            f"""
            QSplitter#classifySplitter {{
                background-color: transparent;
            }}
            QSplitter#classifySplitter::handle {{
                background-color: transparent;
                margin: 0 6px;
            }}
            QSplitter#classifySplitter::handle:hover {{
                background-color: {P.border};
                border-radius: 2px;
            }}
            """
        )

        # ---------- LEFT: scrollable input area ----------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # The scrollbar lives inside the left pane; make it visually subtle.
        left_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )

        left = QWidget()
        left.setObjectName("classifyLeftPane")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(2, 2, 8, 2)   # small right inset so content doesn't hug scrollbar
        left_l.setSpacing(14)

        # Header
        header = QVBoxLayout()
        h_sub = QLabel("Classify")
        h_sub.setObjectName("subtitleLabel")
        h_title = QLabel("Foot image analysis")
        h_title.setObjectName("titleLabel")
        h_hint = QLabel(
            "Upload up to three foot views and (optionally) clinical "
            "measurements. The AI will classify the foot type and recommend "
            "an insole configuration."
        )
        h_hint.setStyleSheet(f"color: {P.text_secondary}; font-size: 12px;")
        h_hint.setWordWrap(True)
        header.addWidget(h_sub)
        header.addWidget(h_title)
        header.addWidget(h_hint)
        left_l.addLayout(header)

        # Image row: three drop zones
        img_row = QHBoxLayout()
        img_row.setSpacing(12)
        self.zone_lateral = ImageDropZone("Lateral view", "Side view of foot")
        self.zone_top     = ImageDropZone("Top view (AP)", "Top / dorsal view")
        self.zone_back    = ImageDropZone("Back view", "Posterior / heel view")
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            img_row.addWidget(z, 1)
            z.image_changed.connect(self._update_cta_state)
        left_l.addLayout(img_row)

        # Measurements
        self.meas_panel = MeasurementPanel()
        self.meas_panel.measurements_changed.connect(self._on_measurements_changed)
        left_l.addWidget(self.meas_panel)

        # Patient code + CTA row
        cta_row = QHBoxLayout()
        cta_row.setSpacing(12)
        patient_label = QLabel("Patient code")
        patient_label.setObjectName("sectionLabel")
        self.patient_code = QLineEdit()
        self.patient_code.setPlaceholderText("e.g. P1097 (optional)")
        self.patient_code.setMaximumWidth(180)
        cta_row.addWidget(patient_label)
        cta_row.addWidget(self.patient_code)
        cta_row.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        cta_row.addWidget(self.clear_btn)

        self.cta = QPushButton("Analyze Foot")
        self.cta.setObjectName("primaryButton")
        self.cta.setEnabled(False)
        self.cta.clicked.connect(self._on_classify)
        cta_row.addWidget(self.cta)

        left_l.addLayout(cta_row)
        # Trailing stretch so content sits at the top, not centered.
        left_l.addStretch(1)

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ---------- RIGHT: results ----------
        self.results = ResultsPanel()
        splitter.addWidget(self.results)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([800, 600])
        outer.addWidget(splitter, 1)

    # -------------------------------------------------------------- events
    def _on_measurements_changed(self, vals: dict) -> None:
        self._measurements = vals
        self._update_cta_state()

    def _update_cta_state(self, *_args) -> None:
        any_image = any(
            z.image_path for z in (self.zone_lateral, self.zone_top, self.zone_back)
        )
        self.cta.setEnabled(any_image)

    def _on_clear(self) -> None:
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            z.clear()
        self.meas_panel.clear()
        self.patient_code.clear()
        self.results.clear()

    def _on_classify(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self.cta.setEnabled(False)
        self.cta.setText("Analyzing…")

        self._worker = InferenceWorker(
            api_base_url=APP_CONFIG.api_base_url,
            lateral_path=self.zone_lateral.image_path,
            top_path=self.zone_top.image_path,
            back_path=self.zone_back.image_path,
            measurements=self._measurements,
            patient_code=(self.patient_code.text().strip() or None),
            use_local_fallback=APP_CONFIG.use_local_inference_fallback,
            parent=self,
        )
        self._worker.finished_ok.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._reset_cta)
        self._worker.start()

    def _on_result(self, result: dict) -> None:
        self.results.set_result(result)

    def _on_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Analysis failed", msg)

    def _reset_cta(self) -> None:
        self._update_cta_state()
        self.cta.setText("Analyze Foot")

    # ------------------------------------------------------------- helpers
    def set_images(self, lateral: Path | None, top: Path | None, back: Path | None) -> None:
        """Programmatic setter — used by demo button / drag from disk."""
        if lateral:
            self.zone_lateral.set_image(lateral)
        if top:
            self.zone_top.set_image(top)
        if back:
            self.zone_back.set_image(back)
