# Insole Foot Classification

> Measurement-first foot-classification system for insole recommendation.
> Deterministic clinical classification, with an assistive image-based
> estimator, a native desktop GUI, a FastAPI service, a PostgreSQL store,
> and Docker orchestration.

---

## What it does

Given an optional set of clinical measurements and up to **three foot
images** (lateral / top / back), the system:

1. **Classifies the foot into one of five clinical categories** — Severe
   Flat Arch, Flat Arch, Normal Foot, High Arch, Severe High Arch —
   defined by the project's arch-height bands (cm):
   Severe Flat `< 2.7` · Flat `2.7–3.5` · Normal `3.6–5.5` ·
   High `5.6–6.4` · Severe High `> 6.4`.
2. Emits a **recommended insole configuration** (arch support height,
   heel cup depth, medial post strength, lateral wedge strength,
   forefoot cushioning) from the generative branch.
3. Persists every result to PostgreSQL with full audit metadata,
   **including which path produced it** (measured vs estimated).

### Two paths, clearly distinguished

| Path | Trigger | Trust | Confidence |
|---|---|---|---|
| **Measured** (authoritative) | Arch-height measurement supplied | Deterministic; exact by construction | 100% |
| **Estimated** (assistive) | No measurement — estimated from images | Approximate; **must be confirmed clinically** | Honest, sub-100% |

The measured path applies the project's arch-height bands directly. It
does **not** consult the neural network's class output, so it is exact by
construction. The estimated path uses the model to estimate arch height
from images and is **always flagged in the UI as non-authoritative**.

> **Why measurement-first?** The original brief targeted ≥90%
> *image-based* accuracy. Empirical diagnostics during development showed
> arch height is **not visually recoverable** from the supplied photo
> views (image-only accuracy ≈ 33%, with 0% recall on Normal and High),
> while a deterministic rule on the measurement scores ≈ 100%. The
> success criterion was formally revised accordingly. See
> `Success_Criterion_Revision.docx` and `Verification_and_Test_Report.docx`
> in the delivery package, both reproducible via the diagnostic scripts.

---

## Architecture

```
                ┌───────────────────────────────────────────┐
                │           Desktop GUI (PySide6)           │
                │  ┌─────────────────┐  ┌───────────────┐   │
                │  │ Classification  │  │   Training    │   │
                │  │      tab        │  │      tab      │   │
                │  └────────┬────────┘  └────────┬──────┘   │
                └───────────┼────────────────────┼──────────┘
                            │ HTTP (Docker backend; local fallback)
                            ▼                              ▼
                ┌───────────────────────────┐   ┌──────────────────────┐
                │   FastAPI service         │   │  Training (threaded, │
                │  /api/classify            │   │  in backend container)│
                │  /api/training/runs       │   └──────────┬───────────┘
                │  /api/data/summary        │              │
                │  /api/patients            │              │
                └─────────┬─────────────────┘              │
                          │                                 │
              ┌───────────┴───────────┐         ┌──────────┴─────────┐
              ▼                       ▼         ▼                    ▼
       ┌─────────────┐        ┌─────────────┐  ┌──────────────┐  ┌────────┐
       │ Predictor   │        │ Repositories│  │ Multi-view   │  │ Data   │
       │ measurement │        │ (SQLAlchemy)│  │ network      │  │ loader │
       │ -first      │        └──────┬──────┘  └──────┬───────┘  └────┬───┘
       └──────┬──────┘               │                │               │
              │ rule on bands        ▼                ▼               ▼
              │ (authoritative) ┌─────────────────────────────────────────┐
       ┌─────────────┐          │              PostgreSQL                 │
       │ Checkpoints │          │  patients · classifications ·           │
       │  (volume)   │          │  measurements · training_runs           │
       └─────────────┘          └─────────────────────────────────────────┘
```

### Model role (honest description)

The network is **not** the classifier for the core deliverable. The
deterministic rule is. The network's verified, value-adding roles are:

- **Image → measurement estimation** — an assistive pre-fill when no
  measurement is supplied (approximate; flagged non-authoritative).
- **Generative insole-config head** — class-conditional insole
  recommendation.

```
   Lateral ─┐
   Top  ────┼─► [ViewEncoder × 3]
   Back ────┘        │
                     ▼
              MultiModalFusion ──► fused embedding ──┬─► measurements_hat (5)
                     ▲                               ├─► insole_config (5)
   Measurements ─────┘                               └─► logits (assistive only)
   (+ mask)
```

The estimated-path classification is computed by applying the same
arch-height rule to the model's *estimated* arch height — never taken
directly from `logits`, which diagnostics showed are unreliable for this
data.

---

## Quick start (Docker — the supported path)

```bash
cp .env.example .env          # set POSTGRES_PASSWORD; POSTGRES_PORT if 5432 is taken

# Build & start db + backend
docker compose up -d --build db backend
docker compose ps             # both should be (healthy)

# Confirm the model loaded a real checkpoint (NOT random weights)
curl http://localhost:8000/api/health    # expect "model_loaded": true

# Launch the desktop GUI
python -m app.main
```

> **Important:** unlike the original design, the system **does not** fall
> back to random weights. If no trained checkpoint is present the
> Predictor raises `NoTrainedModelError` rather than silently producing
> garbage. Ensure `backend/model/checkpoints/best.pt` exists (training
> writes it; it is retained, not renamed away).

### Train (only needed for the estimator / insole head)

```bash
# 1. Place the consolidated measurement workbook
#    -> data/Sheet/measurements_consolidated.xlsx
# 2. ALWAYS verify before training:
docker compose exec backend python scripts/verify_dataset.py
#    expect: 5/5 classes populated, 0 duplicate patients
# 3. Train via the GUI Training tab, or:
docker compose exec backend python scripts/train.py --epochs 50
```

Training is **not required** for the core (measured) classification — that
path is a deterministic rule. Train only to improve the assistive
estimator and the insole-config head.

### Build a standalone `.exe` (Windows)

```bash
pip install pyinstaller
python app/build_exe.py        # -> dist/InsoleFootClassification/
```

The hardened build script verifies a checkpoint exists, confirms the
binary was produced, and bundles `best.pt` so first launch works.

---

## Verification & diagnostics (reproducible)

These scripts are permanent parts of the codebase. They exist so that any
claim about the system can be independently re-checked — a discipline
adopted after early development produced several misleading "100%"
results.

```bash
# Dataset soundness: class balance, view coverage, patient-leakage check
docker compose exec backend python scripts/verify_dataset.py

# Model characterisation: confusion matrix + the
# measured / image-only / measurement-only accuracy decomposition
docker compose exec backend python scripts/diagnose_model.py
```

Expected headline results (held-out split):

| Probe | Accuracy |
|---|---|
| Rule on true measurements (no ML) | ≈ 100% |
| Model with measurements present | ≈ 88% |
| Model images-only | ≈ 33% |

The first row is the delivered behaviour. The third is why image-only
classification is exposed only as an assistive, flagged path.

---

## Project layout

```
insole-foot-classification/
├── app/                                # PySide6 desktop GUI
│   ├── main.py                         # Entry point
│   ├── build_exe.py                    # Hardened PyInstaller packager
│   └── ui/
│       ├── tabs/                       # classification_tab, training_tab
│       ├── widgets/                    # dropzone, measurement_panel,
│       │                               #   results_panel (provenance banner)
│       ├── workers/                    # inference + training QThreads
│       └── theme/
│
├── backend/
│   ├── model/
│   │   ├── config.py                   # CLASS_NAMES, ARCH_HEIGHT_BANDS
│   │   ├── architectures/              # encoders, fusion, VAE, heads
│   │   ├── data/
│   │   │   ├── dataset.py              # multi-sheet consolidation,
│   │   │   │                           #   mm→cm, measurement-derived labels
│   │   │   ├── transforms.py
│   │   │   └── dataloader.py           # stratified split + safe sampler
│   │   ├── training/                   # losses, metrics, trainer
│   │   ├── inference/
│   │   │   └── predictor.py            # MEASUREMENT-FIRST; no random fallback
│   │   └── utils/checkpoint.py         # best.pt-priority selector
│   │
│   ├── database/                       # SQLAlchemy + Alembic + Pydantic
│   │   ├── connection.py               # SA 2.0 text()-safe
│   │   ├── schemas.py                  # + classification_source
│   │   └── repositories/
│   │
│   └── server/                         # FastAPI
│       └── routes/                     # health, classification, training,
│                                       #   data_router, patients
│
├── data/
│   ├── Heel/  Flat/  Normal/  Sheet/   # images by cohort + measurement xlsx
│   └── README.md
│
├── docker/                             # Dockerfile.backend (+trainer),
│                                       #   postgres-init.sql
├── scripts/
│   ├── verify_dataset.py               # RUN BEFORE TRAINING
│   ├── diagnose_model.py               # reproduces the evidence
│   ├── train.py  predict.py  prepare_data.py  seed_demo_data.py
│   └── export_onnx.py
│
├── docker-compose.yml                  # db + backend + trainer + pgadmin
│                                       #   (shm_size 4gb; INSOLE_FORCE_WORKERS=0)
├── alembic.ini  pyproject.toml
├── requirements.txt  .env.example  .gitignore
└── README.md
```

---

## API reference (excerpt)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness + `model_loaded` + DB status |
| `/api/classify` | POST | Multipart: optional images + `measurements_json`; returns class, **`classification_source`** (`measured`/`image_estimated`), probabilities, estimated measurements, insole config |
| `/api/training/runs` | POST | Start a training run (threaded in backend) |
| `/api/training/runs` | GET | List recent runs |
| `/api/training/runs/{id}` | GET | Run status + live per-epoch history |
| `/api/data/summary` | GET | Scan `data/` and report counts |
| `/api/patients` | POST/GET | Patient records |

Swagger UI at `http://localhost:8000/docs`.

---

## Database

Tables: `patients` (by patient code, e.g. `P1097`), `classifications`
(one row per inference, includes the provenance flag), `measurements`,
`training_runs` (full audit trail). Migrations via Alembic.

```bash
docker compose up -d db
docker compose exec backend alembic upgrade head
```

---

## Configuration

Runtime config via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | derived from `POSTGRES_*` | SQLAlchemy DSN |
| `POSTGRES_PORT` | `5432` | Host port (set `5433` if 5432 is in use) |
| `API_PORT` | `8000` | FastAPI port |
| `DATA_DIR` | `/workspace/data` | Dataset root (in container) |
| `DEFAULT_CHECKPOINT_PATH` | `backend/model/checkpoints/best.pt` | Loaded at startup |
| `INSOLE_FORCE_WORKERS` | `0` | DataLoader workers (0 in containers) |

A single configurable knob, `ARCH_HEIGHT_MM_TO_CM` in
`backend/model/data/dataset.py`, controls the millimetre→centimetre
conversion for measurement sheets. Change it only if a cohort is
confirmed to use a different unit.

---

## Known limitations (verified, documented)

- **Image-only classification is unreliable** for this dataset because
  arch height is not visually encoded in the supplied views. It is
  exposed only as an assistive, clearly-flagged estimate.
- **Estimated measurements are approximate** and must be confirmed by a
  clinical measurement before use.
- Class boundaries follow the project's fixed arch-height bands; changing
  them requires a central configuration change.

---

## Confidentiality

Per the project agreement, the dataset, trained checkpoints, and outputs
are **confidential and remain the property of the project owner**. The
`.gitignore` excludes `data/Heel/*`, `data/Flat/*`, `data/Normal/*`,
`data/Sheet/*`, and `backend/model/checkpoints/*.pt`. Do not push data or
trained models to any external or public registry.

---

## License

Proprietary. See the project agreement.
