"""
color_segmentation.py
======================
Algoritmo DETERMINISTICO (color thresholding, nessun ML/DL) per separare,
nelle immagini dei test grafomotori (meandri/spirali) del dataset HandPD,
il tratto NERO di riferimento dal tratto BLU disegnato dal paziente.

Le immagini hanno:
    - sfondo bianco (con leggera tinta/rumore di scansione)
    - tracciato di riferimento NERO
    - tratto del paziente BLU

Il metodo si basa sull'osservazione (verificata empiricamente su un
campione di immagini reali del dataset) che:

    - I pixel BLU hanno il canale B (Blue) nettamente più alto rispetto
      ai canali R e G (differenza tipica > 100), mentre nello sfondo
      bianco questa differenza è piccola (~10-20) anche se lo sfondo ha
      una lieve dominante bluastra dovuta alla scansione.
    - I pixel NERI hanno tutti e tre i canali (B, G, R) bassi e simili
      tra loro (bassa luminosità, bassa "colorazione").
    - Lo sfondo ha invece tutti e tre i canali alti (> 200).

Per questo la separazione viene fatta in due passaggi, in ordine:
    1) Si isola il BLU: differenza tra canale Blue e il massimo tra
       Red/Green superiore a una soglia (blue_diff_thresh).
    2) Si isola il NERO: pixel scuri (max canale < dark_thresh) che
       NON sono già stati classificati come blu.

L'ordine (prima blu, poi nero-esclusivo-blu) evita che i pixel scuri
sul bordo del tratto blu (anti-aliasing/JPEG artifacts) vengano
erroneamente conteggiati come tratto nero.
"""

from __future__ import annotations
import numpy as np
import cv2


def load_image_bgr(image_path: str) -> np.ndarray:
    """
    Carica un'immagine da disco nel formato BGR (convenzione OpenCV).

    Parameters
    ----------
    image_path : str
        Percorso del file immagine (jpg/png/...).

    Returns
    -------
    np.ndarray
        Immagine come array (H, W, 3) in formato BGR, dtype uint8.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {image_path}")
    return img


def extract_blue_black_masks(
    image_bgr: np.ndarray,
    blue_diff_thresh: int = 30,
    dark_thresh: int = 100,
    denoise_kernel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa il tratto BLU e il tratto NERO in due matrici binarie,
    tramite color thresholding puro.

    Parameters
    ----------
    image_bgr : np.ndarray
        Immagine di input in formato BGR (H, W, 3), come restituita da
        cv2.imread / load_image_bgr.
    blue_diff_thresh : int, default=30
        Soglia sulla differenza (B - max(R,G)) oltre la quale un pixel
        viene considerato "blu". Valore stimato empiricamente: nello
        sfondo questa differenza è tipicamente < 20, nel tratto blu
        è tipicamente > 100. 30 è un compromesso prudente che riduce
        falsi positivi/negativi anche in presenza di rumore JPEG.
    dark_thresh : int, default=100
        Soglia sulla luminosità massima (max(R,G,B)) sotto la quale un
        pixel viene considerato "scuro" (candidato tratto nero). Nel
        dataset i pixel del tracciato nero hanno valori tipici < 20,
        lo sfondo > 200: 100 lascia ampio margine di sicurezza.
    denoise_kernel : int, default=0
        Se > 0, applica un'operazione morfologica di apertura (erosione
        + dilatazione) con kernel quadrato di questa dimensione per
        rimuovere pixel isolati (rumore/sale-e-pepe). Con 0 non viene
        applicato alcun post-processing: le maschere sono il risultato
        "puro" del thresholding, come richiesto.

    Returns
    -------
    blue_mask : np.ndarray
        Matrice binaria (H, W), dtype uint8, valori {0, 1}: 1 dove è
        presente il tratto BLU del paziente.
    black_mask : np.ndarray
        Matrice binaria (H, W), dtype uint8, valori {0, 1}: 1 dove è
        presente il tratto NERO di riferimento.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr deve essere un array (H, W, 3) in formato BGR")

    # Lavoriamo in interi con segno per evitare overflow/underflow
    # tipici di uint8 nelle sottrazioni tra canali.
    img = image_bgr.astype(np.int16)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # --- 1) Maschera BLU -----------------------------------------------
    # Un pixel è "blu" se il canale Blue supera nettamente sia Red
    # che Green. Questo isola il tratto del paziente anche quando lo
    # sfondo ha una leggera dominante bluastra dovuta alla scansione.
    blue_diff = B - np.maximum(R, G)
    blue_mask = (blue_diff > blue_diff_thresh).astype(np.uint8)

    # --- 2) Maschera NERA ------------------------------------------------
    # Un pixel è "nero" se è scuro (tutti i canali bassi) e non è già
    # stato classificato come blu (evita di catturare i bordi
    # anti-aliasati/compressione del tratto blu, che risultano più
    # scuri dello sfondo ma appartengono comunque al tratto blu).
    max_channel = np.maximum(np.maximum(R, G), B)
    dark_mask = (max_channel < dark_thresh).astype(np.uint8)
    black_mask = (dark_mask & (1 - blue_mask)).astype(np.uint8)

    # --- 3) Post-processing opzionale (pulizia rumore) -------------------
    if denoise_kernel and denoise_kernel > 0:
        kernel = np.ones((denoise_kernel, denoise_kernel), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    return blue_mask, black_mask


def masks_to_visual(
    image_bgr: np.ndarray, blue_mask: np.ndarray, black_mask: np.ndarray
) -> np.ndarray:
    """
    Crea un'immagine RGB di debug/verifica: sfondo bianco, tratto blu
    colorato in blu puro, tratto nero colorato in nero puro, utile
    per un controllo visivo rapido della qualità della segmentazione.

    Returns
    -------
    np.ndarray
        Immagine (H, W, 3) in formato BGR pronta per essere salvata
        con cv2.imwrite.
    """
    h, w = blue_mask.shape
    visual = np.full((h, w, 3), 255, dtype=np.uint8)  # sfondo bianco
    visual[black_mask == 1] = (0, 0, 0)          # nero (BGR)
    visual[blue_mask == 1] = (255, 0, 0)         # blu puro (BGR)
    return visual


if __name__ == "__main__":
    # Piccolo esempio di utilizzo su un'immagine singola (modificare il path).
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Uso: python color_segmentation.py <percorso_immagine>")
        sys.exit(1)

    image = load_image_bgr(path)
    blue_mask, black_mask = extract_blue_black_masks(image)

    print(f"Immagine: {path}")
    print(f"Shape: {image.shape}")
    print(f"Pixel blu rilevati:  {int(blue_mask.sum())}")
    print(f"Pixel neri rilevati: {int(black_mask.sum())}")
