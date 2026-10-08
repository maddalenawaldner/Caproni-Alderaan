from pathlib import Path
from config import album_structure, path_revisioni


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------FUNZIONI LISTA FILE SELEZIONABILI-----------------------------------------------------------
# -----------------------------------------------------------------(colonna di sinistra)----------------------------------------------------------------

# Crea una lista delle immagini per ogni album e assegna ✓ ai file revisionati
def build_radio_choices(reviewed_set: set, album_name: str) -> list:
    """Scelte radio con prefisso ✓ per le immagini già revisionate."""
    _choices = []
    for fname, fpath in album_structure.get(album_name, []):
        if Path(fpath).stem in reviewed_set:
            _choices.append((f"✓  {fname}", fpath))
        else:
            _choices.append((fname, fpath))
    return _choices


# Ritorna il nome delle immagini appena revisionate (file .jsonl)
def get_reviewed_images() -> set:
    """Immagini che hanno già un file JSONL = già revisionate."""
    if not path_revisioni.exists():
        return set()
    _reviewed = set()
    for f in path_revisioni.iterdir():
        if f.suffix == ".jsonl":
            _reviewed.add(f.stem)
    return _reviewed