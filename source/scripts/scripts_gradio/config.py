from pathlib import Path
import json

# Definizione dei path
path_album            = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
path_tag_dir          = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
path_tag_original_dir = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_original")
path_testo_dir        = Path("/data/caproni/maddalena_waldner/workspace/outputs/Album Trento JPG_split")
path_maschere_dir     = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
path_maschere_singole = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni_unite")
path_revisioni        = Path("/data/caproni/maddalena_waldner/workspace/outputs/gradio/revisioni")
path_revisioni.mkdir(parents=True, exist_ok=True)
path_dizionario_tag = Path("/data/caproni/maddalena_waldner/workspace/outputs/tag_en_ita.json")
TAG_EN_ITA = json.loads(path_dizionario_tag.read_text(encoding="utf-8")) if path_dizionario_tag.exists() else {}

# Crea una lista ordinata alfabeticamente di tutti i path delle img
_imgs_non_ordinate = []
for album in path_album.iterdir():
    for img in album.iterdir():
        _imgs_non_ordinate.append(img)
lista_path_img = sorted(_imgs_non_ordinate)

# Crea un dizionario {nome album: [(nome img, stringa path)]} e una lista con i nomi degli album ordinata alfabeticamente
album_structure = {}
for _album in sorted(path_album.iterdir()):
    _files_album = []
    for f in sorted(_album.iterdir()):
        _files_album.append((f.name, str(f)))
    album_structure[_album.name] = _files_album
album_names_sorted = list(album_structure.keys())