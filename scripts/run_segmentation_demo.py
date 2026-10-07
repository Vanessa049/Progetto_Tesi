"""
run_segmentation_demo.py
=========================
Script dimostrativo: applica l'algoritmo di color thresholding
(src/color_segmentation.py) alle immagini del dataset HandPD e salva
UNICAMENTE le due matrici binarie richieste, come file .npy:
    - <nome>_blue_mask.npy   (matrice 2D, 1=tratto blu, 0=altrove)
    - <nome>_black_mask.npy  (matrice 2D, 1=tratto nero, 0=altrove)

Pensato per la seguente struttura di cartelle (cartella principale HANDPD/):

    HANDPD/
    ├── Meander_HandPD.csv
    ├── Spiral_HandPD.csv
    ├── SpiralHandPD/
    │   ├── SpiralPatients/
    │   └── SpiralControl/
    ├── MeanderHandPD/
    │   ├── MeanderPatients/
    │   └── MeanderControl/
    ├── src/
    │   └── color_segmentation.py
    ├── scripts/
    │   └── run_segmentation_demo.py   <- questo file
    └── outputs/
        ├── masks/    <- qui vengono salvati i file .npy (questo script)
        └── plots/    <- qui vengono salvate le figure di plot_masks.py

Esecuzione da terminale (dalla cartella principale HANDPD/):
    python scripts/run_segmentation_demo.py --image "MeanderHandPD/MeanderPatients/0002-5.jpg"

oppure, senza --image, lo script processa automaticamente N immagini
di esempio prese da SpiralHandPD/ e MeanderHandPD/ (utile come smoke test),
oppure l'intero dataset con --n-samples -1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Permette di importare src/color_segmentation.py eseguendo lo script
# sia da terminale sia da VS Code, indipendentemente dalla cwd.
ROOT_DIR = Path(__file__).resolve().parent.parent  # cartella HANDPD/
sys.path.append(str(ROOT_DIR / "src"))

from color_segmentation import load_image_bgr, extract_blue_black_masks  # noqa: E402

# Sottocartelle in cui cercare le immagini, relative alla root HANDPD/.
# Vengono elencate più varianti di nome (con/senza underscore) per
# adattarsi automaticamente a piccole differenze nella struttura locale.
DEFAULT_SEARCH_DIRS = [
    "MeanderHandPD", "Meander_HandPD",
    "SpiralHandPD", "Spiral_HandPD",
]


def process_single_image(image_path: Path, output_dir: Path) -> None:
    """Esegue la segmentazione su una singola immagine e salva i risultati."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    image = load_image_bgr(str(image_path))
    blue_mask, black_mask = extract_blue_black_masks(image)

    # Salvataggio delle due matrici binarie 2D richieste (uniche output)
    np.save(output_dir / f"{stem}_blue_mask.npy", blue_mask)
    np.save(output_dir / f"{stem}_black_mask.npy", black_mask)

    print(
        f"[OK] {image_path.name}: "
        f"pixel blu={int(blue_mask.sum())}, pixel neri={int(black_mask.sum())} "
        f"-> risultati salvati in {output_dir}"
    )


def find_dataset_images(root_dir: Path) -> list[Path]:
    """Cerca tutte le immagini nelle sottocartelle standard MeanderHandPD/SpiralHandPD."""
    images: list[Path] = []
    for sub in DEFAULT_SEARCH_DIRS:
        d = root_dir / sub
        if d.exists():
            images.extend(sorted(d.rglob("*.jpg")))
            images.extend(sorted(d.rglob("*.png")))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo segmentazione colore (blu/nero) su immagini HandPD."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Percorso di una singola immagine da processare.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR),
        help="Cartella radice del dataset (default: cartella HANDPD/, un livello sopra scripts/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "masks"),
        help="Cartella dove salvare i risultati (default: outputs/masks/).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Numero di immagini di esempio da processare se --image non è specificato. "
             "Usa -1 per processare l'intero dataset.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.image:
        process_single_image(Path(args.image), output_dir)
        return

    data_dir = Path(args.data_dir)
    images = find_dataset_images(data_dir)
    if not images:
        print(
            f"Nessuna immagine trovata in {data_dir}/MeanderHandPD o {data_dir}/SpiralHandPD. "
            f"Verifica la struttura delle cartelle oppure usa --image / --data-dir."
        )
        return

    selected = images if args.n_samples == -1 else images[: args.n_samples]
    print(f"Trovate {len(images)} immagini totali, ne processo {len(selected)}.")
    for image_path in selected:
        process_single_image(image_path, output_dir)


if __name__ == "__main__":
    main()