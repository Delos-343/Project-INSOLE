"""
Classification tab (revision 2026-05).

Changes:
  * Manual measurement input panel REMOVED (revision B3). Measurements are
    never typed — they are looked up from the consolidated sheet by
    patient code on the backend.
  * Patient code is now a primary, prominent input (it is the lookup key).
  * Left pane is image upload + patient code + actions only.
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
from app.ui.widgets.results_panel import ResultsPanel
from app.ui.workers.inference_worker import InferenceWorker


class ClassificationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._build()

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("classifySplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(16)
        splitter.setStyleSheet(
            f"""
            QSplitter#classifySplitter::handle {{ background: transparent; margin: 0 6px; }}
            QSplitter#classifySplitter::handle:hover {{
                background-color: {P.border}; border-radius: 2px; }}
            """
        )

        # ---------- LEFT ----------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(2, 2, 8, 2)
        left_l.setSpacing(14)

        sub = QLabel("Classify")
        sub.setObjectName("subtitleLabel")
        title = QLabel("Patient Analysis")
        title.setObjectName("titleLabel")
        hint = QLabel(
            "Enter the patient code and upload the foot views. Clinical "
            "measurements are retrieved automatically from the consolidated "
            "records."
        )
        hint.setStyleSheet(f"color:{P.text_secondary};font-size:12px;")
        hint.setWordWrap(True)
        left_l.addWidget(sub)
        left_l.addWidget(title)
        left_l.addWidget(hint)

        # Patient code — now primary
        pc_box = QFrame()
        pc_box.setStyleSheet(
            f"QFrame{{background:{P.bg_secondary};border:1px solid {P.border};"
            f"border-radius:10px;}}"
        )
        pc_l = QHBoxLayout(pc_box)
        pc_l.setContentsMargins(14, 14, 14, 14)
        pc_l.setSpacing(10)
        pc_lbl = QLabel("Patient code")
        pc_lbl.setStyleSheet(
            f"color:{P.text_primary};font-size:13px;font-weight:600;"
        )
        self.patient_code = QLineEdit()
        self.patient_code.setPlaceholderText("e.g. P014  (used to look up measurements)")
        self.patient_code.setClearButtonEnabled(True)
        self.patient_code.returnPressed.connect(self._on_classify)
        pc_l.addWidget(pc_lbl)
        pc_l.addWidget(self.patient_code, 1)
        left_l.addWidget(pc_box)

        # Image row
        img_row = QHBoxLayout()
        img_row.setSpacing(12)
        self.zone_lateral = ImageDropZone("Lateral view", "Side view of foot")
        self.zone_top = ImageDropZone("Top view (AP)", "Top / dorsal view")
        self.zone_back = ImageDropZone("Back view", "Posterior / heel view")
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            img_row.addWidget(z, 1)
        left_l.addLayout(img_row)

        # Actions
        act = QHBoxLayout()
        act.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        act.addWidget(self.clear_btn)
        self.cta = QPushButton("Analyze Foot")
        self.cta.setObjectName("primaryButton")
        self.cta.clicked.connect(self._on_classify)
        act.addWidget(self.cta)
        left_l.addLayout(act)
        left_l.addStretch(1)

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ---------- RIGHT ----------
        self.results = ResultsPanel()
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([720, 680])
        outer.addWidget(splitter, 1)

    # ----------------------------------------------------------- events
    def _on_clear(self) -> None:
        for z in (self.zone_lateral, self.zone_top, self.zone_back):
            z.clear()
        self.patient_code.clear()
        self.results.clear()

    def _on_classify(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        code = self.patient_code.text().strip()
        if not code:
            QMessageBox.information(
                self, "Patient code required",
                "Enter the patient code so measurements can be retrieved "
                "from the consolidated records.",
            )
            return
        self.cta.setEnabled(False)
        self.cta.setText("Analyzing…")
        self._worker = InferenceWorker(
            api_base_url=APP_CONFIG.api_base_url,
            lateral_path=self.zone_lateral.image_path,
            top_path=self.zone_top.image_path,
            back_path=self.zone_back.image_path,
            measurements=None,                       # never sent from GUI now
            patient_code=code,
            use_local_fallback=APP_CONFIG.use_local_inference_fallback,
            parent=self,
        )
        self._worker.finished_ok.connect(self.results.set_result)
        self._worker.failed.connect(
            lambda m: QMessageBox.warning(self, "Analysis failed", m)
        )
        self._worker.finished.connect(self._reset_cta)
        self._worker.start()

    def _reset_cta(self) -> None:
        self.cta.setEnabled(True)
        self.cta.setText("Analyze Foot")
