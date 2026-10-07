"""
plot_masks.py
==============
Controllo qualità VISIVO delle matrici binarie generate da
color_segmentation.py, utile per valutare l'attendibilità dei dati
prima di usarli per addestrare un altro modello.

Per ogni immagine controllata, mostra 4 pannelli affiancati:
    1) Immagine originale
    2) Matrice blue_mask (bianco/nero: bianco = tratto blu rilevato)
    3) Matrice black_mask (bianco/nero: bianco = tratto nero rilevato)
    4) Overlay di verifica: sfondo bianco, tratto blu rilevato colorato
       di blu, tratto nero rilevato colorato di nero -> permette di
       confrontare a colpo d'occhio il risultato con l'originale

Due modalità di esecuzione (dalla cartella HANDPD/):

    # una singola immagine specifica
    python scripts/plot_masks.py --image "Meander_HandPD/MeanderPatients/0002-5.jpg"

    # N immagini scelte a caso tra TUTTO il dataset (meandri + spirali,
    # pazienti + controlli). Ad ogni esecuzione il campione cambia,
    # utile per un controllo qualità esteso nel tempo senza rivedere
    # sempre le stesse immagini.
    python scripts/plot_masks.py --n-random 8

Le figure vengono salvate come .png in outputs/plots/, così puoi
riguardarle o allegarle alla tesi senza dover rilanciare lo script.
Con --image la figura viene anche mostrata a schermo; con --n-random
le figure vengono solo salvate (per non dover chiudere N finestre una
per una), e alla fine viene stampato l'elenco dei file generati.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from color_segmentation import (  # noqa: E402
    load_image_bgr,
    extract_blue_black_masks,
    masks_to_visual,
)

DEFAULT_SEARCH_DIRS = [
    "MeanderHandPD", "Meander_HandPD",
    "SpiralHandPD", "Spiral_HandPD",
]


def find_dataset_images(root_dir: Path) -> list[Path]:
    images: list[Path] = []
    for sub in DEFAULT_SEARCH_DIRS:
        d = root_dir / sub
        if d.exists():
            images.extend(sorted(d.rglob("*.jpg")))
            images.extend(sorted(d.rglob("*.png")))
    return images


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """matplotlib si aspetta RGB, OpenCV lavora in BGR: invertiamo i canali."""
    return img[:, :, ::-1]


def build_quality_check_figure(image_path: Path):
    """Costruisce la figura a 4 pannelli per una immagine (senza mostrarla)."""
    image = load_image_bgr(str(image_path))
    blue_mask, black_mask = extract_blue_black_masks(image)
    overlay = masks_to_visual(image, blue_mask, black_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Controllo qualità: {image_path.name}")

    axes[0].imshow(bgr_to_rgb(image))
    axes[0].set_title("Originale")

    axes[1].imshow(blue_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"blue_mask (1={int(blue_mask.sum())} px)")

    axes[2].imshow(black_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"black_mask (1={int(black_mask.sum())} px)")

    axes[3].imshow(bgr_to_rgb(overlay))
    axes[3].set_title("Ricostruzione da maschere")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_single_image(image_path: Path, output_dir: Path) -> None:
    """Costruisce, mostra a schermo e salva la figura di controllo per una
    singola immagine."""
    fig = build_quality_check_figure(image_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_plot.png"
    fig.savefig(out_path, dpi=120)
    print(f"Figura salvata in: {out_path}")

    plt.show()


def plot_n_random(root_dir: Path, output_dir: Path, n: int) -> None:
    """Sceglie N immagini a caso da TUTTO il dataset (meandri + spirali) e
    genera per ciascuna la figura di controllo a 4 pannelli, salvandola
    su disco. Il campione cambia ad ogni esecuzione (nessun seed fisso),
    utile per controlli qualità ripetuti su porzioni diverse del dataset."""
    images = find_dataset_images(root_dir)
    if not images:
        print(f"Nessuna immagine trovata in {root_dir}.")
        return

    n = min(n, len(images))
    sample = random.sample(images, n)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for image_path in sample:
        fig = build_quality_check_figure(image_path)
        out_path = output_dir / f"{image_path.stem}_plot.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"[OK] {image_path.name} -> {out_path}")

    print(f"\n{len(saved_paths)} figure salvate in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot di controllo qualità per le maschere binarie generate."
    )
    parser.add_argument("--image", type=str, default=None,
                         help="Percorso di una singola immagine da controllare.")
    parser.add_argument("--n-random", type=int, default=None,
                         help="Numero di immagini scelte a caso (meandri + spirali) "
                              "da controllare con la vista a 4 pannelli.")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR),
                         help="Cartella radice del dataset (default: HANDPD/).")
    parser.add_argument("--output-dir", type=str,
                         default=str(ROOT_DIR / "outputs" / "plots"),
                         help="Cartella dove salvare le figure generate.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.image:
        plot_single_image(Path(args.image), output_dir)
    elif args.n_random:
        plot_n_random(Path(args.data_dir), output_dir, args.n_random)
    else:
        print("Specifica --image <percorso> oppure --n-random <numero>.")


if __name__ == "__main__":
    main()