from pathlib import Path
from config import path_testo_dir, path_tag_dir, path_maschere_dir
from functions_mask_tags import filtra_tag
from functions_jsonl_file import load_current_revision


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------FUNZIONI CARICA IMMAGINE------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Ritorna il path di un'immagine, dato il nome dell'immagine
def find_mask_path(name_img: str) -> Path | None:
    for album in path_maschere_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                return file
    return None


# Ritorna tutto ciò che è associato ad un'immagine (nome, ocr, tag, maschere)
def load_image_data(img_path: str) -> tuple:
    name_img = Path(img_path).stem
    txt, tags = "", []
    for album in path_testo_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                txt = file.read_text(encoding="utf-8")
    has_text = txt.strip() != "NO_TEXT"
    if not has_text:
        txt = ""
    for album in path_tag_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                raw = file.read_text(encoding="utf-8")
                tags = []
                for t in raw.split(","):
                    t = t.strip()
                    if t:
                        tags.append(t)
    tag_sopravvissuti = filtra_tag(name_img)
    has_masks = bool(tag_sopravvissuti)
    tags = tag_sopravvissuti if tag_sopravvissuti else []
    saved = load_current_revision(name_img)
    if saved:
        txt  = saved.get("ocr", txt)
        tags = saved.get("tag", tags)
    mask = find_mask_path(name_img)
    return (str(mask) if mask else None), name_img, txt, tags, has_masks, has_text