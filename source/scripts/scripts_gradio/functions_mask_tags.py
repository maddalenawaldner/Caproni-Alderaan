import base64
import io
import random
from pathlib import Path
import numpy as np
from PIL import Image
from config import path_tag_dir, path_maschere_singole, album_structure, TAG_EN_ITA


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------FUNZIONI MASCHERE TAG------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

MASK_COLORS = [
    (220,  60,  60, 140),   # rosso
    ( 60, 120, 220, 140),   # blu
    ( 50, 190,  80, 140),   # verde
    (230, 160,   0, 140),   # ambra
    (160,  60, 220, 140),   # viola
    (  0, 190, 190, 140),   # teal
    (230, 100,   0, 140),   # arancione
    (220,  60, 180, 140),   # rosa
    ( 80, 200, 240, 140),   # celeste
    (150, 220,  80, 140),   # lime
]

# Trova il path della cartella con le predizioni di Sam3
def find_pred_dir(name_img: str) -> Path | None:
    if not path_maschere_singole.exists():
        return None
    d = path_maschere_singole / name_img
    if d.is_dir():
        return d
    for d in path_maschere_singole.iterdir():
        if d.is_dir() and Path(d.name).stem == name_img:
            return d
    return None


# Ritorna un dizionario tag_inglese - tag_italiano
def load_tag_en_to_ita(name_img: str) -> dict:
    """Restituisce il dizionario {tag_inglese: tag_italiano}."""
    return TAG_EN_ITA


# Ritorna un dizionario tag_italiano - path maschera png
def load_masks_for_image(name_img: str) -> dict:
    masks = {}
    pred_dir = find_pred_dir(name_img)
    if pred_dir is None:
        return masks
    agg_dir = pred_dir / "masks_aggregated"
    if not agg_dir.is_dir():
        return masks
    en_to_ita = load_tag_en_to_ita(name_img)
    prefix = pred_dir.name + "_"
    for mask_file in sorted(agg_dir.iterdir()):
        if mask_file.suffix.lower() != ".png":
            continue
        en_tag = mask_file.stem
        if en_tag.startswith(prefix):
            en_tag = en_tag[len(prefix):]
        if en_tag:
            ita_tag = en_to_ita.get(en_tag, en_tag)
            masks[ita_tag] = mask_file
    return masks

_overlay_cache: dict = {}
_b64_cache: dict = {}


# Ritorna una lista dei soli tag di cui Sam3 ha prodotto delle maschere aggregate
def filtra_tag(name_img):
    """Restituisce la lista dei tag italiani per cui SAM3 ha prodotto una maschera aggregata."""
    en_to_ita = load_tag_en_to_ita(name_img)
    lista_tag_sopravvissuti = []
    for album in path_maschere_singole.iterdir():
        if album.name == name_img:
            for cartella in album.iterdir():
                if cartella.name == "masks_aggregated":
                    for file in sorted(cartella.iterdir()):
                        if file.suffix.lower() == ".png":
                            en_tag = file.stem[len(name_img) + 1:]
                            lista_tag_sopravvissuti.append(en_to_ita.get(en_tag, en_tag))
    return lista_tag_sopravvissuti


# Ritorna un dizionario nome_tag - maschera overlay sottoforma di matrice NumPy3D
def get_mask_overlays(name_img: str, orig_path: str) -> dict:
    """Carica (o recupera dalla cache) gli overlay RGBA pre-calcolati."""
    if name_img in _overlay_cache:
        return _overlay_cache[name_img]
    mask_paths = load_masks_for_image(name_img)
    if not mask_paths:
        _overlay_cache[name_img] = {}
        return {}
    try:
        w, h = Image.open(orig_path).size
    except Exception:
        return {}
    overlays = {}
    for i, (tag, path) in enumerate(sorted(mask_paths.items())):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        try:
            mask_arr = np.array(
                Image.open(path).convert("L").resize((w, h), Image.LANCZOS)
            )
            ov = np.zeros((h, w, 4), dtype=np.uint8)
            ov[mask_arr > 128] = color
            overlays[tag] = ov
        except Exception:
            pass
    _overlay_cache[name_img] = overlays
    return overlays


# Converte le maschere overlay in formato base64 (serve per poter essere messe su html)
def get_overlay_b64(name_img: str, orig_path: str) -> dict:
    """Converte gli overlay in stringhe base64 PNG (cache separata)."""
    if name_img in _b64_cache:
        return _b64_cache[name_img]
    overlays = get_mask_overlays(name_img, orig_path)
    b64 = {}
    for tag, arr in overlays.items():
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, "PNG", compress_level=1)
        b64[tag] = base64.b64encode(buf.getvalue()).decode()
    _b64_cache[name_img] = b64
    return b64


# Mostra immagine, maschere overlays e le etichette per selezionare/deselezionare le maschere
def make_mask_viewer_html(orig_path: str, overlays_b64: dict) -> str:
    """Viewer con overlay toggleabili via CSS :has() — nessun JavaScript."""
    ext = Path(orig_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with open(orig_path, "rb") as fh:
        orig_b64 = base64.b64encode(fh.read()).decode()

    if not overlays_b64:
        return (
            f'<img src="data:{mime};base64,{orig_b64}" '
            f'style="max-width:100%;max-height:70vh;display:block;margin:0 auto;border-radius:10px;">'
        )

    uid = random.randint(100000, 999999)
    tags_sorted = sorted(overlays_b64.keys())
    n = len(tags_sorted)

    css_lines = [f'#mv_{uid} input[type="checkbox"]{{display:none;}}']
    for i in range(n):
        css_lines.append(
            f'#mv_{uid}:has(#cb_{uid}_{i}:not(:checked)) [data-mv="{uid}_{i}"]'
            f'{{display:none!important;}}'
        )
        css_lines.append(
            f'#mv_{uid}:has(#cb_{uid}_{i}:not(:checked)) label[for="cb_{uid}_{i}"]'
            f'{{opacity:0.38;}}'
        )

    cbs = "".join(f'<input type="checkbox" id="cb_{uid}_{i}" checked>' for i in range(n))

    ov_imgs = []
    for i, tag in enumerate(tags_sorted):
        ov_imgs.append(
            f'<img data-mv="{uid}_{i}" src="data:image/png;base64,{overlays_b64[tag]}" '
            f'style="position:absolute;top:0;left:0;width:100%;height:100%;">'
        )

    labels = []
    for i, tag in enumerate(tags_sorted):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        dot = f"rgb({color[0]},{color[1]},{color[2]})"
        labels.append(
            f'<label for="cb_{uid}_{i}" '
            f'style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#f1f5f9;border:1px solid #cbd5e1;border-radius:20px;'
            f'padding:5px 14px;cursor:pointer;font-size:1rem;'
            f'font-family:Inter,system-ui,sans-serif;user-select:none;transition:opacity 0.15s;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{dot};flex-shrink:0;"></span>'
            f'{tag}</label>'
        )

    css = " ".join(css_lines)
    return (
        f'<div id="mv_{uid}">'
        f'<style>{css}</style>'
        f'{cbs}'
        f'<div style="position:relative;width:fit-content;max-width:100%;margin:0 auto;">'
        f'<img src="data:{mime};base64,{orig_b64}" style="max-width:100%;max-height:70vh;display:block;border-radius:10px;">'
        f'{"".join(ov_imgs)}'
        f'</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;">'
        f'{"".join(labels)}'
        f'</div>'
        f'</div>'
    )


# Ritorna il nome dell'album dato il nome dell'immagine
def find_album_for_image(img_path: str) -> str | None:
    for aname, files in album_structure.items():
        for _, fpath in files:
            if fpath == img_path:
                return aname
    return None


