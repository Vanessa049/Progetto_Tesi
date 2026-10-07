"""
check_masks.py
================
Script di verifica: controlla che le matrici binarie salvate in outputs/masks/
rispettino le proprietà attese:
    - contengono solo valori 0 e 1 (nessun altro valore)
    - hanno la stessa shape (H, W) dell'immagine originale
    - blue_mask e black_mask non si sovrappongono mai (nessun pixel è
      contemporaneamente 1 in entrambe)

Esecuzione (dalla cartella HANDPD/):
    python scripts/check_masks.py
    python scripts/check_masks.py --outputs-dir outputs/masks
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_single_pair(blue_path: Path, black_path: Path) -> list[str]:
    """Controlla una coppia di file _blue_mask.npy / _black_mask.npy e
    restituisce una lista di eventuali problemi trovati (vuota se tutto ok)."""
    problems = []

    blue_mask = np.load(blue_path)
    black_mask = np.load(black_path)

    # 1) Solo valori 0/1
    blue_values = set(np.unique(blue_mask).tolist())
    black_values = set(np.unique(black_mask).tolist())
    if not blue_values.issubset({0, 1}):
        problems.append(f"blue_mask contiene valori non binari: {blue_values}")
    if not black_values.issubset({0, 1}):
        problems.append(f"black_mask contiene valori non binari: {black_values}")

    # 2) Stessa forma (2D, stesse dimensioni tra loro)
    if blue_mask.ndim != 2:
        problems.append(f"blue_mask non è 2D, ha shape {blue_mask.shape}")
    if black_mask.ndim != 2:
        problems.append(f"black_mask non è 2D, ha shape {black_mask.shape}")
    if blue_mask.shape != black_mask.shape:
        problems.append(
            f"shape diverse tra blue_mask {blue_mask.shape} e black_mask {black_mask.shape}"
        )

    # 3) Nessuna sovrapposizione blu/nero
    overlap = int(np.logical_and(blue_mask == 1, black_mask == 1).sum())
    if overlap > 0:
        problems.append(f"{overlap} pixel risultano 1 sia in blue_mask che in black_mask")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica le matrici binarie generate.")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "masks"),
        help="Cartella contenente i file *_blue_mask.npy / *_black_mask.npy",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    blue_files = sorted(outputs_dir.glob("*_blue_mask.npy"))

    if not blue_files:
        print(f"Nessun file *_blue_mask.npy trovato in {outputs_dir}. "
              f"Esegui prima run_segmentation_demo.py.")
        sys.exit(1)

    total = 0
    total_problems = 0
    for blue_path in blue_files:
        stem = blue_path.name.replace("_blue_mask.npy", "")
        black_path = outputs_dir / f"{stem}_black_mask.npy"
        if not black_path.exists():
            print(f"[MANCA] {stem}: trovato blue_mask ma non black_mask corrispondente")
            total_problems += 1
            continue

        total += 1
        problems = check_single_pair(blue_path, black_path)
        if problems:
            total_problems += 1
            print(f"[PROBLEMA] {stem}:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[OK] {stem}: binaria, shape corretta, nessuna sovrapposizione")

    print()
    print(f"Controllate {total} coppie di matrici. Problemi trovati: {total_problems}.")


if __name__ == "__main__":
    main()