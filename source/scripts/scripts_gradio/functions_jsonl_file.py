import json
from datetime import datetime
from pathlib import Path
from config import path_revisioni, path_testo_dir, path_tag_dir
from functions_mask_tags import filtra_tag


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------FUNZIONI .jsonl FILES------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Ritorna il path del file revisionato 
def get_jsonl_path(name_img: str) -> Path:
    return path_revisioni / f"{name_img}.jsonl"


# Ritorna, se c'è, la versione "corrente"
def load_current_revision(name_img: str) -> dict | None:
    p = get_jsonl_path(name_img)
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for line in reversed(lines):
        entry = json.loads(line)
        if entry.get("versione") == "corrente":
            return entry
    return None


# Salva OCT, tags, user, data e ora e mantiene al massimo due versioni
def save_revision(name_img: str, img_path: str, mask_path, txt: str, tags: list, username: str):
    """Mantiene al massimo 2 righe: 'precedente' + 'corrente'."""
    now = datetime.now()
    if isinstance(tags, list):
        _final_tags = tags
    else:
        _final_tags = []
        for t in tags.split(","):
            t = t.strip()
            if t:
                _final_tags.append(t)
    new_entry = {
        "versione":      "corrente",
        "nome_file":     name_img,
        "path_immagine": str(img_path),
        "path_maschere": str(mask_path) if mask_path else "",
        "ocr":           txt,
        "tag":           _final_tags,
        "data_modifica": now.strftime("%Y-%m-%d"),
        "ora_modifica":  now.strftime("%H:%M:%S"),
        "utente":        username,
    }
    old = load_current_revision(name_img)
    entries = []
    if old:
        entries.append(dict(old, versione="precedente"))
    entries.append(new_entry)
    with get_jsonl_path(name_img).open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# Ritorna gli output originali del modello
def load_annulla_target(name_img: str) -> tuple:
    orig_txt, orig_tags = "", []
    for album in path_testo_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                orig_txt = file.read_text(encoding="utf-8")
    for album in path_tag_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                raw = file.read_text(encoding="utf-8")
                orig_tags = []
                for t in raw.split(","):
                    t = t.strip()
                    if t:
                        orig_tags.append(t)
    tag_sopravvissuti = filtra_tag(name_img)
    if tag_sopravvissuti:
        orig_tags = tag_sopravvissuti
    return orig_txt, orig_tags