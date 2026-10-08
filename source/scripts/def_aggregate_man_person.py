"""
Post-processing delle maschere aggregate di SAM3.

Legge le maschere aggregate prodotte da SAM3 (senza modificarle) e scrive in una
cartella separata una copia completa in cui i tag in TAG_ALIASES sono unificati
(OR logico), es. "man" + "person" -> "person".

Input  (solo lettura): path_predizioni/<nome_img>/masks_aggregated/<nome_img>_<tag>.png
Output:                path_output/<nome_img>/masks_aggregated/<nome_img>_<tag_finale>.png

L'output ha la stessa struttura dell'input, quindi l'app Gradio può usarlo
cambiando solo path_maschere_singole.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

path_predizioni = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni")
path_output     = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni_unite")

# Tag da unificare: {tag_originale: tag_finale}
TAG_ALIASES = {
    "man": "person",
}

# True = mostra solo cosa verrebbe fatto, senza scrivere nulla
DRY_RUN = False


def normalize_tag(tag):
    """Riporta il tag alla sua forma canonica (es. "man" -> "person")."""
    return TAG_ALIASES.get(tag.strip().lower(), tag)


def processa_immagine(agg_dir, out_dir, prefix):
    """Scrive in out_dir le maschere di agg_dir con i tag unificati. Ritorna il numero di unioni fatte."""
    # Raggruppa i file per tag finale: {tag_finale: [(tag_originale, path)]}
    gruppi = {}
    for f in sorted(agg_dir.glob("*.png")):
        if not f.stem.startswith(prefix + "_"):
            continue
        tag = f.stem[len(prefix) + 1:]
        gruppi.setdefault(normalize_tag(tag), []).append((tag, f))

    if not DRY_RUN:
        # Riparte da zero per questa immagine, così non restano file di esecuzioni precedenti
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    n_unioni = 0
    for tag_finale, files in gruppi.items():
        out_path = out_dir / f"{prefix}_{tag_finale}.png"

        # Un solo file con il nome già giusto: copia invariata
        if len(files) == 1 and files[0][0] == tag_finale:
            if not DRY_RUN:
                shutil.copy2(files[0][1], out_path)
            continue

        unione = None
        for _, f in files:
            mask = np.array(Image.open(f).convert("L")) > 128
            if unione is not None and mask.shape != unione.shape:
                raise ValueError(f"Dimensioni diverse tra le maschere di {agg_dir}: {f.name}")
            unione = mask if unione is None else (unione | mask)

        nomi = ", ".join(f.name for _, f in files)
        print(f"  {nomi}  ->  {out_path.name}")
        if not DRY_RUN:
            Image.fromarray(unione.astype(np.uint8) * 255).save(out_path)
        n_unioni += 1
    return n_unioni


def main():
    if path_output.resolve() == path_predizioni.resolve():
        raise ValueError("path_output deve essere diverso da path_predizioni")
    if DRY_RUN:
        print("DRY RUN: nessun file verrà scritto. Imposta DRY_RUN = False per applicare.\n")

    n_immagini, n_unioni = 0, 0
    for img_dir in sorted(path_predizioni.iterdir()):
        agg_dir = img_dir / "masks_aggregated"
        if not agg_dir.is_dir():
            continue
        out_dir = path_output / img_dir.name / "masks_aggregated"
        n = processa_immagine(agg_dir, out_dir, img_dir.name)
        if n:
            print(f"↳ {img_dir.name}: {n} unione/i\n")
        n_immagini += 1
        n_unioni += n

    print(f"Immagini elaborate: {n_immagini} — unioni: {n_unioni}")
    if not DRY_RUN:
        print(f"Output salvato in: {path_output}")


if __name__ == "__main__":
    # Uso opzionale: python unisci_tag_maschere.py <cartella_input> <cartella_output>
    if len(sys.argv) > 2:
        path_predizioni = Path(sys.argv[1])
        path_output     = Path(sys.argv[2])
    main()
