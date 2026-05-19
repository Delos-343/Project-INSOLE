"""
ResultsPanel (revision 2026-05, spacing pass 2).

Fixes the two remaining issues visible in the rendered window:

  1. Rows inside group boxes were inheriting the panel's dark fill, so each
     label/value row painted its own rectangle and the rows looked like
     cramped stacked sub-cards. Every inner row container is now an
     explicitly transparent QWidget with its own vertical padding, so the
     group box reads as one clean card with airy rows.

  2. Group-box titles sat almost on top of the preceding section. The
     inter-section gap and the group box's reserved title margin are both
     increased so each title clearly belongs to its own card with space
     above it.

Behaviour (SHEET / BOUNDARY / ESTIMATED provenance, dual-rule display,
always-populated measurements) is unchanged.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme.colors import PALETTE as P

# ---- spacing system ----
PANEL_PAD = 28
SECTION_GAP = 30        # ↑ between major blocks / group boxes
GROUP_TOP = 34          # ↑ inside a group box, below its title
GROUP_SIDE = 20
GROUP_BOTTOM = 20
ROW_VPAD = 7            # vertical padding inside every label/value row


def _sev_color(b: str) -> str:
    return {"normal": P.severity_normal, "moderate": P.severity_moderate,
            "severe": P.severity_severe}.get(b, P.text_muted)


def _groupbox_qss() -> str:
    # 22px top margin reserves clear space for the title ABOVE the frame so
    # it never collides with the previous section.
    return (
        f"QGroupBox {{"
        f"  border: 1px solid {P.border};"
        f"  border-radius: 10px;"
        f"  margin-top: 22px;"
        f"  background-color: {P.bg_secondary};"
        f"}}"
        f"QGroupBox::title {{"
        f"  subcontrol-origin: margin;"
        f"  subcontrol-position: top left;"
        f"  left: 16px;"
        f"  padding: 0 6px;"
        f"  color: {P.text_secondary};"
        f"  font-size: 12px;"
        f"  font-weight: 600;"
        f"}}"
    )


def _row() -> tuple[QWidget, QHBoxLayout]:
    """A transparent row container with its own vertical padding so rows
    breathe and never paint their own background rectangle."""
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, ROW_VPAD, 0, ROW_VPAD)
    lay.setSpacing(12)
    return w, lay


class ResultsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsPanel")
        self.setStyleSheet(
            f"#resultsPanel{{background-color:{P.bg_secondary};"
            f"border:1px solid {P.border};border-radius:12px;}}"
        )
        self._build()
        self.clear()

    def _build(self):
        o = QVBoxLayout(self)
        o.setContentsMargins(PANEL_PAD, PANEL_PAD, PANEL_PAD, PANEL_PAD)
        o.setSpacing(SECTION_GAP)

        t = QLabel("RESULTS")
        t.setStyleSheet(
            f"color:{P.text_muted};font-size:11px;font-weight:700;"
            f"letter-spacing:2px;background:transparent;"
        )
        o.addWidget(t)

        self.banner = QLabel("—")
        self.banner.setWordWrap(True)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setMinimumHeight(56)
        o.addWidget(self.banner)

        self.headline = QLabel("—")
        self.headline.setStyleSheet(
            f"color:{P.text_primary};font-size:26px;font-weight:700;"
            f"background:transparent;"
        )
        o.addWidget(self.headline)

        chips = QWidget()
        chips.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(chips)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)
        self.sev_chip = QLabel("—")
        self.sev_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sev_chip.setMinimumWidth(96)
        self.sev_chip.setStyleSheet(self._chip(P.text_muted))
        cl.addWidget(self.sev_chip)
        self.conf = QLabel("Confidence —")
        self.conf.setStyleSheet(
            f"color:{P.text_secondary};font-size:13px;background:transparent;"
        )
        cl.addWidget(self.conf)
        cl.addStretch()
        o.addWidget(chips)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        b = QVBoxLayout(body)
        b.setContentsMargins(0, 0, 8, 0)
        b.setSpacing(SECTION_GAP)

        # --- Classification rules ---
        self.rules_box = QGroupBox("Classification rules")
        self.rules_box.setStyleSheet(_groupbox_qss())
        rl = QVBoxLayout(self.rules_box)
        rl.setContentsMargins(GROUP_SIDE, GROUP_TOP, GROUP_SIDE, GROUP_BOTTOM)
        rl.setSpacing(0)
        self.arch_lbl = QLabel("Arch-height rule: —")
        self.heel_lbl = QLabel("Heel-angle rule: —")
        for w in (self.arch_lbl, self.heel_lbl):
            rw, rlay = _row()
            w.setStyleSheet(
                f"color:{P.text_secondary};font-size:12px;background:transparent;"
            )
            w.setWordWrap(True)
            rlay.addWidget(w)
            rl.addWidget(rw)
        b.addWidget(self.rules_box)

        # --- Class probabilities ---
        pb = QGroupBox("Class probabilities")
        pb.setStyleSheet(_groupbox_qss())
        pl = QVBoxLayout(pb)
        pl.setContentsMargins(GROUP_SIDE, GROUP_TOP, GROUP_SIDE, GROUP_BOTTOM)
        pl.setSpacing(0)
        self.prob_rows = {}
        for c in ("Severe Flat Arch", "Flat Arch", "Normal Foot",
                  "High Arch", "Severe High Arch"):
            rw, r = _row()
            n = QLabel(c)
            n.setMinimumWidth(140)
            n.setStyleSheet(
                f"color:{P.text_secondary};font-size:12px;background:transparent;"
            )
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            v = QLabel("0.0%")
            v.setMinimumWidth(52)
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v.setStyleSheet(
                f"color:{P.text_primary};font-size:12px;background:transparent;"
            )
            r.addWidget(n)
            r.addWidget(bar, 1)
            r.addWidget(v)
            pl.addWidget(rw)
            self.prob_rows[c] = (bar, v)
        b.addWidget(pb)

        # --- Measurements ---
        self.meas_box = QGroupBox("Measurements")
        self.meas_box.setStyleSheet(_groupbox_qss())
        ml = QVBoxLayout(self.meas_box)
        ml.setContentsMargins(GROUP_SIDE, GROUP_TOP, GROUP_SIDE, GROUP_BOTTOM)
        ml.setSpacing(0)
        self.meas_labels = {}
        self._mk = [
            ("calcaneal_inclination_deg", "Calcaneal inclination", "°"),
            ("heel_angle_deg", "Heel angle", "°"),
            ("arch_height_cm", "Arch height", "cm"),
            ("kite_angle_deg", "Kite angle", "°"),
            ("first_metatarsal_talus_deg", "1st metatarsal–talus", "°"),
        ]
        for k, lbl, unit in self._mk:
            rw, row = _row()
            a = QLabel(lbl)
            a.setStyleSheet(
                f"color:{P.text_secondary};font-size:12px;background:transparent;"
            )
            val = QLabel(f"— {unit}")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            val.setStyleSheet(
                f"color:{P.text_primary};font-size:13px;font-weight:600;"
                f"background:transparent;"
            )
            self.meas_labels[k] = val
            row.addWidget(a)
            row.addStretch()
            row.addWidget(val)
            ml.addWidget(rw)
        b.addWidget(self.meas_box)

        # --- Insole config ---
        ib = QGroupBox("Recommended insole configuration")
        ib.setStyleSheet(_groupbox_qss())
        il = QVBoxLayout(ib)
        il.setContentsMargins(GROUP_SIDE, GROUP_TOP, GROUP_SIDE, GROUP_BOTTOM)
        il.setSpacing(0)
        self.insole_rows = {}
        for k, lbl in [
            ("arch_support_height", "Arch support height"),
            ("heel_cup_depth", "Heel cup depth"),
            ("medial_post_strength", "Medial post strength"),
            ("lateral_wedge_strength", "Lateral wedge strength"),
            ("forefoot_cushioning", "Forefoot cushioning"),
        ]:
            rw, r = _row()
            a = QLabel(lbl)
            a.setMinimumWidth(170)
            a.setStyleSheet(
                f"color:{P.text_secondary};font-size:12px;background:transparent;"
            )
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            r.addWidget(a)
            r.addWidget(bar, 1)
            il.addWidget(rw)
            self.insole_rows[k] = bar
        b.addWidget(ib)

        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet(
            f"color:{P.text_muted};font-size:11px;font-style:italic;"
            f"padding-top:6px;background:transparent;"
        )
        b.addWidget(self.notes)
        b.addStretch()

        scroll.setWidget(body)
        o.addWidget(scroll, 1)

    def _chip(self, c):
        return (f"background-color:{c}22;color:{c};font-weight:700;"
                f"font-size:10px;letter-spacing:2px;text-transform:uppercase;"
                f"border:1px solid {c}66;border-radius:999px;padding:6px 14px;")

    def _ban(self, c):
        return (f"background-color:{c}1F;color:{c};border:1px solid {c}80;"
                f"border-radius:8px;padding:14px 18px;font-size:12px;"
                f"font-weight:600;")

    def clear(self):
        self.banner.setText("Awaiting input")
        self.banner.setStyleSheet(self._ban(P.text_muted))
        self.headline.setText("—")
        self.sev_chip.setText("—")
        self.sev_chip.setStyleSheet(self._chip(P.text_muted))
        self.conf.setText("Confidence —")
        self.arch_lbl.setText("Arch-height rule: —")
        self.heel_lbl.setText("Heel-angle rule: —")
        for bar, v in self.prob_rows.values():
            bar.setValue(0)
            v.setText("0.0%")
        for v in self.meas_labels.values():
            v.setText("—")
        for bar in self.insole_rows.values():
            bar.setValue(0)
        self.notes.setText("")

    def set_result(self, r: dict):
        src = r.get("classification_source", "sheet")
        cls = r.get("predicted_class", "—")
        conf = float(r.get("confidence", 0.0))
        sev = r.get("severity_band", "unknown")
        agree = r.get("rules_agree", True)

        if src == "sheet" and agree:
            self.banner.setText(
                "✓  SHEET — authoritative result.\n"
                "Measurements retrieved from the consolidated records; "
                "deterministic dual-rule classification."
            )
            self.banner.setStyleSheet(self._ban(P.success))
            self.meas_box.setTitle("Measurements (from consolidated records)")
        elif src == "sheet" and not agree:
            self.banner.setText(
                "⚑  BOUNDARY.\n\n"
                "Found in records, but the arch-height and heel-angle rules "
                "disagree by one class. Headline is the authoritative "
                "arch-height class."
            )
            self.banner.setStyleSheet(self._ban(P.accent))
            self.meas_box.setTitle("Measurements (from consolidated records)")
        else:
            self.banner.setText(
                "⚠  ESTIMATED — assistive only, NOT authoritative.\n"
                "Patient not in the consolidated records; arch height "
                "estimated from images. Confirm before use."
            )
            self.banner.setStyleSheet(self._ban(P.warning))
            self.meas_box.setTitle("Measurements (model-estimated)")

        self.headline.setText(cls)
        self.conf.setText(f"Confidence {conf*100:0.1f}%")
        col = _sev_color(sev)
        self.sev_chip.setText(sev)
        self.sev_chip.setStyleSheet(self._chip(col))

        ac = r.get("arch_class")
        hc = r.get("heel_class")
        self.arch_lbl.setText(f"Arch-height rule (authoritative): {ac or '—'}")
        self.heel_lbl.setText(f"Heel-angle rule (corroborating): {hc or 'n/a'}")
        self.heel_lbl.setStyleSheet(
            f"color:{P.danger if (hc and not agree) else P.text_secondary};"
            f"font-size:12px;background:transparent;"
        )

        probs = r.get("class_probabilities") or {}
        for name, (bar, v) in self.prob_rows.items():
            p = float(probs.get(name, 0.0))
            bar.setValue(int(round(p*100)))
            v.setText(f"{p*100:0.1f}%")
            chunk = col if name == cls else P.accent_muted
            bar.setStyleSheet(
                f"QProgressBar{{background-color:{P.bg_tertiary};border:none;"
                f"border-radius:3px;}}"
                f"QProgressBar::chunk{{background-color:{chunk};"
                f"border-radius:3px;}}"
            )

        units = {"calcaneal_inclination_deg": "°", "heel_angle_deg": "°",
                 "arch_height_cm": "cm", "kite_angle_deg": "°",
                 "first_metatarsal_talus_deg": "°"}
        # The API serialises the looked-up/estimated values as
        # 'measurements_predicted'. Older paths used 'measurements_used'.
        # Read whichever is populated so a key rename on either side
        # cannot silently blank this card again.
        used = (
            r.get("measurements_predicted")
            or r.get("measurements_used")
            or r.get("measurements_estimated")
            or {}
        )
        tag = "" if src == "sheet" else " (est.)"
        for k, val_lbl in self.meas_labels.items():
            unit = units[k]
            if k in used and used[k] is not None:
                val_lbl.setText(f"{float(used[k]):.2f} {unit}{tag}")
            else:
                val_lbl.setText(f"— {unit}")

        insole = r.get("insole_configuration") or {}
        for k, bar in self.insole_rows.items():
            bar.setValue(int(round(float(insole.get(k, 0.0))*100)))

        notes = r.get("notes") or []
        # Blank line between bullets so multi-note results read as
        # separate paragraphs rather than one wrapped block.
        self.notes.setText("• " + "\n\n• ".join(notes) if notes else "")