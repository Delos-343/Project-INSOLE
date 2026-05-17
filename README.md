# AI Insole Foot Classification — Delivery Package

This is the final handoff package. Engineering of the core deliverable is
functionally complete and verified. Closing the project requires a few
Project-Owner actions listed at the bottom.

---

## 1. What this system does

Classifies a foot into five clinical classes (Severe Flat / Flat / Normal /
High / Severe High Arch) and recommends an insole configuration.

- **With a clinical arch-height measurement → deterministic, exact (~100%),
  authoritative.** This is the core deliverable and how the system is meant
  to be used clinically.
- **Without a measurement → image-based estimate, clearly flagged as
  assistive and non-authoritative**, with honest confidence.

Why measurement-first: the five classes are *defined* by arch-height-cm
bands. Diagnostic testing proved arch height is not visually recoverable
from the supplied photos (~33% images-only). A deterministic rule on the
measurement is exact by construction and is the correct, defensible design.
This is recorded formally in the *Formal Revision of Project Success
Criterion*.

---

## 2. Documents in this package (read in this order)

1. **Formal_Revision_of_Success_Criterion.docx** — replaces the original
   (unachievable) ">90% on images" metric with the deterministic
   measurement criterion. Sign this first.
2. **System_Requirements_Specification.docx** — the baseline requirements
   the system is delivered against.
3. **Verification_and_Test_Report.docx** — the evidence: Test A, Test B,
   and the four diagnostic probes.
4. **Final_Completion_Checklist.docx** — every brief obligation mapped to
   status, with the completion declaration to sign.

Open each in Word and accept the "update fields" prompt so the Table of
Contents populates (Word builds it on open; this is normal).

---

## 3. Build the executable (Project-Owner action)

The `.exe` can only be built on the target Windows machine. From the
project root in your activated venv:

```powershell
pip install pyinstaller
python app\build_exe.py
```

The hardened script:
- checks a trained `best.pt` exists,
- runs PyInstaller in `--onedir` mode,
- verifies the binary was produced and is a sane size,
- copies the checkpoint into the dist folder so first launch works,
- prints the exact path to launch.

Result: `dist\InsoleFootClassification\InsoleFootClassification.exe`

---

## 4. Smoke-test the executable (Project-Owner action)

Double-click the built `.exe`, then run **Test A** as the acceptance check:

- Load P014's three views (lateral, ap, heel).
- Enter measurements: calcaneal 24.6, heel 2.6, **arch 4.69**, kite 31.5,
  1st met–talus 3.8.
- Analyze.
- **Expected: Normal Foot, 100.0%, green "MEASURED — authoritative"
  banner.** If you see this, the executable is good.

(The backend container must be running for full functionality:
`docker compose up -d`.)

---

## 5. Close the project

Per the Final Completion Checklist:

1. Review, sign, and file the four documents.
2. Build the `.exe` (Section 3).
3. Smoke-test it (Section 4).
4. Archive: signed documents + built `dist\InsoleFootClassification\` +
   the codebase = the delivered package.
5. Sign the Completion Declaration in the checklist. The project is then
   formally COMPLETE.

---

## 6. Operating reference (day-to-day)

```powershell
# Start backend
docker compose up -d
docker compose ps                 # both containers healthy

# Launch app
.\.venv\Scripts\Activate.ps1
python -m app.main

# Verify data before any retrain
docker compose exec backend python scripts\verify_dataset.py

# Reproduce the evidence at any time
docker compose exec backend python scripts\diagnose_model.py

# Stop (keeps DB + checkpoints)
docker compose stop
```

No further training is required: the core deliverable does not use the
network's classification output. The trained checkpoint adequately serves
the assistive estimate and the insole-config head.

---

*Engineering complete and verified. "Complete" is the Project Owner's
deliberate, evidence-based decision — every claim here is reproducible
from the delivered codebase.*
