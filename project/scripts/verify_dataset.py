"""
Pre-training verification — run this AFTER placing the consolidated sheet in
data/Sheet/ and BEFORE retraining. Confirms the new measurement-driven
labelling produces a sane, 5-class, image-matched dataset.

    docker compose exec backend python scripts/verify_dataset.py
or locally:
    python scripts/verify_dataset.py
"""

from __future__ import annotations

from collections import Counter

from backend.model.data.dataset import FootClassificationDataset


def main() -> None:
    print("Scanning data/ with measurement-driven labels...\n")
    ds = FootClassificationDataset("data", transform=None)

    n = len(ds)
    print(f"Total matched samples (image + measurement): {n}")
    if n == 0:
        print(
            "\n[FAIL] Zero samples. Check that:\n"
            "  1. data/Sheet/ contains the consolidated .xlsx\n"
            "  2. data/Normal, data/Flat, data/Heel contain P#### folders\n"
        )
        return

    print("\nClass distribution:")
    for cls, c in ds.class_counts().items():
        bar = "#" * int(60 * c / max(1, n))
        print(f"  {cls:<18} {c:>5}  {bar}")

    # View coverage.
    v = Counter()
    for s in ds.samples:
        v["lateral"] += s.lateral_path is not None
        v["top"] += s.top_path is not None
        v["back"] += s.back_path is not None
    print("\nView coverage:")
    for k in ("lateral", "top", "back"):
        print(f"  {k:<8} {v[k]:>5} / {n}  ({100*v[k]//max(1,n)}%)")

    # Leakage check.
    ids = [s.patient_id for s in ds.samples]
    dup = len(ids) - len(set(ids))
    print(f"\nUnique patients: {len(set(ids))}  | duplicate IDs: {dup}")

    # Sanity: are there >=2 non-empty classes?
    populated = sum(1 for c in ds.class_counts().values() if c > 0)
    print(f"Populated classes: {populated} / 5")

    print("\n" + ("[OK] Dataset looks healthy — safe to retrain."
                   if populated >= 4 and n > 100 and dup == 0
                   else "[WARN] Review the numbers above before training."))


if __name__ == "__main__":
    main()