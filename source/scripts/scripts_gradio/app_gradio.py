import base64
import io
import json
import random
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image
import numpy as np

# Definizione dei path
path_album        = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
path_tag_dir      = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
path_tag  = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
path_testo_dir    = Path("/data/caproni/maddalena_waldner/workspace/outputs/Album Trento JPG_split")
path_maschere_dir     = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
path_maschere_singole = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni")
path_revisioni    = Path("/data/caproni/maddalena_waldner/workspace/outputs/gradio/revisioni")
path_revisioni.mkdir(parents=True, exist_ok=True)

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
    album_structure[_album.name] = _files_album    # aggiunge al dizionario il nome dell'album (chiave) e la lista di tuple nome file-path (valore)
album_names_sorted = list(album_structure.keys())



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------FUNZIONI LISTA FILE SELEZIONABILI-----------------------------------------------------------
# -----------------------------------------------------------------(colonna di sinistra)----------------------------------------------------------------


# Costruisce la lista di scelte per il radio button di un album, aggiungendo ✓ ai file già revisionati
def build_radio_choices(reviewed_set: set, album_name: str) -> list:    # viene specificato di che tipo devono essere gli arggomenti passati e che la funzione restituirà una lista
    """Scelte radio con prefisso ✓ per le immagini già revisionate."""
    _choices = []
    for fname, fpath in album_structure.get(album_name, []):   # album structure è la variabile dizionario creata fuori     # .get restituisce il valore corrispondente se album_name esiste nel dizionario, altrimenti resituisce una lista vuota
        if Path(fpath).stem in reviewed_set:
            _choices.append((f"✓  {fname}", fpath))
        else:
            _choices.append((fname, fpath))
    return _choices                                            # restituisce una lista di tuple di tutti i file (nome file, path) con l'aggiunta di ✓ davanti al nome se il file è stato revisionato


# Legge la cartella revisioni e restituisce un set con i nomi dei file già revisionati
def get_reviewed_images() -> set:
    """Immagini che hanno già un file JSONL = già revisionate."""
    if not path_revisioni.exists():                            # evita un errore nel caso la cartella delle revisioni non esistesse
        return set()
    _reviewed = set()
    for f in path_revisioni.iterdir():
        if f.suffix == ".jsonl":
            _reviewed.add(f.stem)
    return _reviewed                                           # restituisce un set dei file già revisionati (se non ce ne sono restituisce un set vuoto)


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------FUNZIONI MASCHERE TAG------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Colori RGBA per sovrapporre le maschere (R, G, B, alpha 0-255)
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

# Restituisce la cartella predizioni/<nome_img>/ (accetta sia lo stem sia il nome con estensione)
def find_pred_dir(name_img: str) -> Path | None:
    if not path_maschere_singole.exists():
        return None
    d = path_maschere_singole / name_img
    if d.is_dir():
        return d
    for d in path_maschere_singole.iterdir():                 # fallback: cartella chiamata es. "foto_01.jpg"
        if d.is_dir() and Path(d.name).stem == name_img:
            return d
    return None


def load_tag_en_to_ita(name_img: str) -> dict:
    """Restituisce {tag_inglese: tag_italiano} per un'immagine."""
    en_tags, ita_tags = [], []
    for album in path_tag_dir.iterdir():
        for f in album.iterdir():
            if f.stem == name_img:
                en_tags = [t.strip() for t in f.read_text(encoding="utf-8").split(",") if t.strip()]
    for album in path_tag.iterdir():
        for f in album.iterdir():
            if f.stem == name_img:
                ita_tags = [t.strip() for t in f.read_text(encoding="utf-8").split(",") if t.strip()]
    return dict(zip(en_tags, ita_tags))


# Restituisce {tag_italiano: Path} leggendo le maschere aggregate (una per tag) da predizioni/<img>/masks_aggregated/
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
            en_tag = en_tag[len(prefix):]                     # rimuove "nome_img_" e tiene solo il tag inglese
        if en_tag:
            ita_tag = en_to_ita.get(en_tag, en_tag)          # fallback all'inglese se manca la traduzione
            masks[ita_tag] = mask_file
    return masks

# Cache numpy overlay arrays: {name_img → {tag → RGBA np.ndarray}}
_overlay_cache: dict = {}                                              # contiene i dati grezzi dell'immagine della maschera convertiti in array multidimensionali NumPy (np.ndarray) con formato colore RGBA (Rosso, Verde, Blu e Trasparenza/Alpha); serve per fare operazioni veloci senza dover rileggere ogni volta i dati
# Cache base64 PNG strings degli overlay: {name_img → {tag → str}}
_b64_cache: dict = {}                                                  # contiene le stesse maschere convertite in stringhe di testo formattate in Base64; anche questo serve per questioni di velocità
                                                                       # N.B. Entrambi i dizionari contengono la stessa informazione (nel primo le maschere sono sottoforma di matrice NumPy Array; nel secondo sottoforma di stringa)


# Restituisce la lista dei tag italiani per cui SAM3 ha prodotto una maschera aggregata ([] se l'immagine non è stata elaborata)
def filtra_tag(name_img):
    en_to_ita = load_tag_en_to_ita(name_img)
    lista_tag_sopravvissuti = []
    for album in path_maschere_singole.iterdir():
        if album.name == name_img:
            for cartella in album.iterdir():
                if cartella.name == "masks_aggregated":
                    for file in sorted(cartella.iterdir()):
                        if file.suffix.lower() == ".png":
                            en_tag = file.stem[len(name_img) + 1:]    # toglie "nome_img_" e tiene il tag inglese
                            lista_tag_sopravvissuti.append(en_to_ita.get(en_tag, en_tag))
    return lista_tag_sopravvissuti



def get_mask_overlays(name_img: str, orig_path: str) -> dict:
    """Carica (o recupera dalla cache) gli overlay RGBA pre-calcolati."""
    if name_img in _overlay_cache:
        return _overlay_cache[name_img]                       # se nel dict c'è il nome dell'immagine ritorna il valore associato (quindi le maschere sono già state calcolate e le restituisce subito)
    mask_paths = load_masks_for_image(name_img)               # recupera il dizionario dei percorsi delle maschere tag-path
    if not mask_paths:
        _overlay_cache[name_img] = {}
        return {}                                             # se non c'è il dizionario assegna alla chiave un dzionario vuoto e ritorna un dizionario vuoto
    try:                                                      # try e except vengono utilizzati assieme: try enta di eseguire un'azione "rischiosa" e except definisce cosa fare se qualcosa va storto durante l'esecuzione - Lo scopo di queste righe è trovare la larghezza e l'altezza in pixel dell'immagine originale di sfondo
        w, h = Image.open(orig_path).size                     # viene aperta l'immagine e .size restituisce le dimensioni
    except Exception:
        return {}                                             # se si verifica un qualsiasi errore nel try passa all'except e la funzione restituisce zero overlay (dizionario vuoto)
    overlays = {}                                             # dizionario che raccoglie le maschere dell'immagine corrente
    for i, (tag, path) in enumerate(sorted(mask_paths.items())):      # ordina alfabeticamnete le coppie tag-path e assegna ad i un numero e a tag e path il rispettivo tag e path
        color = MASK_COLORS[i % len(MASK_COLORS)]                     # assegna un colore ad ogni maschera: enumera con i tutte le maschere e assegna il colore 0 se i % numero dei colori è 0, ecc ecc.. (questo fa sì che se le maschere superano il numero dei colori disponibili si riparte da capo)
        try:
            mask_arr = np.array(                                                     # dopo aver fatto il passaggio nella riga sotto converte l'immagine in una matrice bidimensionale
                Image.open(path).convert("L").resize((w, h), Image.LANCZOS)          # apre l'immagine della maschera, la converte in scala di grigli ("L"), la ridimensiona per farla coincidere con w e h calcolate sopra. l'algoritmo LANCZOS è un filtro di alta qualità per preservare i dettagli dei bordi durante il ridimensionamento
            )
            ov = np.zeros((h, w, 4), dtype=np.uint8)                                 # crea una matrice tridimensionale di soli zero (nera e trasparente al 100%)
            ov[mask_arr > 128] = color                                               # crea una matrice booleana di valori True o False: sarà True dove la maschera è bianca o chiara (sopra 128 di grigio) e False dove è nera o scura (sotto 128); poi applica il colore assegnato solo ed esclusivamente nei punti in cui la condizione è True
            overlays[tag] = ov                                                       # salva la matrice RGBA appena colorata nel dizionario overlays usando come chiave il nome del tag
        except Exception:
            pass
    _overlay_cache[name_img] = overlays                                              # assegna il dizionario appena creato alla chiave "nome dell'immagine" nel dict _overlay_cache
    return overlays                                                                  # restituisce il dizionario nome tag (chiave) e np.ndarray, ossia la matrice 3D in formato RGBA (valore)


def get_overlay_b64(name_img: str, orig_path: str) -> dict:                          # prende le matrici NumPy generate da get_mask_overlays e le converte in stringhe Base64 codificate in formato PNG (serve per inviare le maschere direttamente all'interfaccia grafica senza doverle salvare su disco)
    """Converte gli overlay in stringhe base64 PNG (cache separata)."""
    if name_img in _b64_cache:
        return _b64_cache[name_img]
    overlays = get_mask_overlays(name_img, orig_path)
    b64 = {}
    for tag, arr in overlays.items():
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, "PNG", compress_level=1)
        b64[tag] = base64.b64encode(buf.getvalue()).decode()
    _b64_cache[name_img] = b64                                                       # assegna il dizionario nome tag - stringa alla chiave "nome immagine" del dizionario _b64_cache
    return b64                                                                       # ritorna il dizionario nome tag - stringa


def make_mask_viewer_html(orig_path: str, overlays_b64: dict) -> str:                            # genera un componente HTML/CSS autonomo (viewer) che mostra l'immagine originale e vi sovrappone le maschere colorate che è possibile attivare e disattivare
    """Viewer con overlay toggleabili via CSS :has() — nessun JavaScript."""
    ext = Path(orig_path).suffix.lower().lstrip(".")                                             # restituisce il suffisso senza . (es. jpg)
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with open(orig_path, "rb") as fh:
        orig_b64 = base64.b64encode(fh.read()).decode()                                          # l'immagine viene convertita in una stringa base64

    if not overlays_b64:                                                                         # se non ci sono maschere restituisce l'immagine originale e si ferma
        return (
            f'<img src="data:{mime};base64,{orig_b64}" '
            f'style="max-width:100%;max-height:70vh;display:block;margin:0 auto;border-radius:10px;">'   # altezza massima 70% della finestra, centrata
        )

    uid = random.randint(100000, 999999)                                                         # viene generato un numero casuale per evitare conflitti se più immagini vengono visualizzate nella stessa pagina
    tags_sorted = sorted(overlays_b64.keys())
    n = len(tags_sorted)

    # Regole CSS: :has() nasconde overlay e dimma label quando checkbox è deselezionata          #
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

    # Checkbox nascoste (lo stato è gestito dal CSS, non dal JS)
    cbs = "".join(f'<input type="checkbox" id="cb_{uid}_{i}" checked>' for i in range(n))

    # Overlay images
    ov_imgs = []
    for i, tag in enumerate(tags_sorted):
        ov_imgs.append(
            f'<img data-mv="{uid}_{i}" src="data:image/png;base64,{overlays_b64[tag]}" '
            f'style="position:absolute;top:0;left:0;width:100%;height:100%;">'
        )

    # Label-pill cliccabili che attivano/disattivano la checkbox corrispondente
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
        f'<div style="position:relative;width:fit-content;max-width:100%;margin:0 auto;">'       # il contenitore si adatta all'immagine, così le maschere restano allineate
        f'<img src="data:{mime};base64,{orig_b64}" style="max-width:100%;max-height:70vh;display:block;border-radius:10px;">'
        f'{"".join(ov_imgs)}'
        f'</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;">'
        f'{"".join(labels)}'
        f'</div>'
        f'</div>'
    )


# Restituisce il nome dell'album dal path di un'immagine
def find_album_for_image(img_path: str) -> str | None:
    for aname, files in album_structure.items():
        for _, fpath in files:
            if fpath == img_path:
                return aname
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------FUNZIONI .jsonl FILES------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Restituisce il percorso del file .jsonl per un'immagine
def get_jsonl_path(name_img: str) -> Path:
    return path_revisioni / f"{name_img}.jsonl"                # path_revisioni è già un path e / serve per concatenare percosi, aggiungendo un pezzetto alla fine


# Legge il file .jsonl e restituisce l'ultima entry con versione = "corrente" oppure None se non esiste
def load_current_revision(name_img: str) -> dict | None:       # ci dice che la funzione restituisce o un dizionario oppure niente
    p = get_jsonl_path(name_img)
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]            # legge il file jsonl, lo divide per righe strippate e appende ciascuna riga alla lista
    for line in reversed(lines):                 # inverte l'ordine della lista perchè l'ultima riga del file .jsonl è sempre la più recente
        entry = json.loads(line)                 # prende la riga del jsonl (è una stringa) e la converte in un dizionario. n.b. .loads è un metodo apposito per i file json
        if entry.get("versione") == "corrente":  # controlla se la chiave "versione" = "corrente"
            return entry
    return None                                  # ritorna la versione corrente sottoforma di dizionario oppure niente se non esiste il file json


# Salva la revisione: se ne esiste una corrente la degrada a "precedente" e salva la nuova come "corrente" (massimo 2 righe)
def save_revision(name_img: str, img_path: str, mask_path, txt: str, tags: list, username: str):
    """Mantiene al massimo 2 righe: 'precedente' + 'corrente'."""
    now = datetime.now()
    if isinstance(tags, list):                                # isinstance verifica il tipo di una variabile (variabile, tipo)
        _final_tags = tags
    else:
        _final_tags = []
        for t in tags.split(","):
            t = t.strip()
            if t:
                _final_tags.append(t)                         # se tags è già una lista la tiene così com'è; se è una stringa la trasforma in lista
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
    }                                                         # crea una nuova entrata (sottoforma di dizionario) del file jsonl con tutte le informazioni
    old = load_current_revision(name_img)                     # recupera la versione corrente e la assegan ad old
    entries = []
    if old:
        entries.append(dict(old, versione="precedente"))      # se old esiste (potrebbe essere None se non c'è alcuna versione corrente) prende il dizionario ed assegna a "versione" il valore "precedente"; aggiunge old alla lista come primo elemento
    entries.append(new_entry)                                 # aggiunge la nuova entrata alla lista (se esiste old nella lista, la new entry è il secondo elemento della lista)
    with get_jsonl_path(name_img).open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n") # .dumps trasforma il dizionario in una stringa e ensure_ascii=False conserva i caratteri speciali (es. lettere accentate o ✓); la stringa viene scritta nel jsonl con un a capo


# Restituisce i valori a cui "Annulla" deve tornare:
# "precedente" dal JSONL se esiste, altrimenti l'output originale del modello
def load_annulla_target(name_img: str) -> tuple:
    orig_txt, orig_tags = "", []                              # recupera testo e tag previsti dal modello
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
    return orig_txt, orig_tags                                # ritorna gli output originali



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------FUNZIONI CARICA IMMAGINE------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# Data un'immagine, restituisce il path della maschera corrispondente
def find_mask_path(name_img: str) -> Path | None:
    for album in path_maschere_dir.iterdir():
        for file in album.iterdir():
            if file.stem == name_img:
                return file
    return None                                # dato il nome di un file ritorna il path delle maschere corrispondenti oppure None se non trova il nome uguale


# Dato il path di un'immagine, restituisce testo OCR e tag; se esiste una revisione salvata sovrascrive con i valori modificati
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
                        tags.append(t)                           # assegna a txt e tags il testo e i tag originali estratti dal modello
    tag_sopravvissuti = filtra_tag(name_img)
    has_masks = bool(tag_sopravvissuti)
    tags = tag_sopravvissuti if tag_sopravvissuti else []         # se non ci sono maschere, nessun tag viene mostrato
    saved = load_current_revision(name_img)                      # assegna a saved il dizionario con i valori della versione corrente (None se non esiste il file jsonl)
    if saved:
        txt  = saved.get("ocr", txt)
        tags = saved.get("tag", tags)
    mask = find_mask_path(name_img)
    return (str(mask) if mask else None), name_img, txt, tags, has_masks, has_text





# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------CSS------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# è la sezione che definisce l'aspetto visivo di tutta l'interfaccia (html definisce i contenuti e la struttura; css definisce l'estetica, quindi colori di sfondo, font, grandezza del testo) - è strutturato a cascata, ossia le regole più specifiche o lette per ultime sovrascrivono quelle generali precedenti

custom_css = """
/* Carica il font Inter da google fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Definisce lo sfondo della pagina e lo applica su tutti i contenitori principali. !important serve a sovrascrivere i valori di default di Gradio */
body, .gradio-container, .main, .wrap {
    background: #f0f2f7 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Elimina in fondo alla pagina la scritta "buit with gradio" */
footer, .footer, div.footer, .built-with,
.svelte-byatnx, .gradio-footer {
    display: none !important;
}

/* Definisce la dimensione dentro a tutti i gradio-container */
.gradio-container, .gradio-container * {
    font-size: 1.15rem !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
label, .label-wrap span {
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
textarea, input, .input-wrap input {
    font-size: 1.15rem !important;
}

/* Contenitore principale trasparente e centrato */
#login_card {
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    max-width: 580px;
    margin: 8vh auto 0;
}

/* Stile del rettangolo interno: bordo elegante, stondato e con padding generale */
#login_card > .form,
#login_card .group,
#login_card [data-testid="block"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.08) !important;
    overflow: hidden !important;
    padding: 24px !important;                 /* Aggiunge spazio uniforme lungo tutto il bordo interno */
}

/* Distanza tra l'header ("Progetto Caproni...") e i campi da compilare */
#login_card .prose,
#login_card [data-testid="markdown"] {
    margin-bottom: 24px !important;          /* Aumenta lo spazio sotto l'intestazione */
}

/* Sposta il pulsante "Accedi" più in basso rispetto ai campi di testo */
#login_card button {
    margin-top: 20px !important;             /* Crea distacco dai campi sopra */
    padding: 12px !important;                 /* Rende il pulsante un po' più alto ed elegante */
}

/* Pulizia bordi interni dei fieldset */
#login_card .fieldset {
    border: none !important;
}


/* Definisce la colonna di sinistra con gli album */
#album_col {
    background: #1e2d4a !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(15,23,42,0.18) !important;
    border: none !important;
    padding: 16px 12px !important;
    min-width: 220px !important;
    max-width: 320px !important;
    align-self: flex-start !important;
    margin-right: 20px !important;

    /* La colonna non supera l'altezza della finestra e scorre per conto suo */
    position: sticky !important;
    top: 12px !important;
    max-height: calc(100vh - 24px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    flex-wrap: nowrap !important;
}

/* Impedisce che gli album chiusi vengano schiacciati quando se ne apre uno */
#album_col > * {
    flex-shrink: 0 !important;
}


/* Scrollbar sottile e in tinta con lo sfondo scuro */
#album_col::-webkit-scrollbar { width: 8px; }
#album_col::-webkit-scrollbar-thumb { background: #3b4f75; border-radius: 8px; }
#album_col::-webkit-scrollbar-track { background: transparent; }
#album_col { scrollbar-width: thin; scrollbar-color: #3b4f75 transparent; }

#album_col h3, #album_col .prose h3 {
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}

#album_col .accordion {
    background: #253655 !important;
    border: none !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}

/* 1. NOMI DEGLI ALBUM (Album 1, Album 2, ecc.) */
#album_col button,
#album_col .accordion button,
#album_col .accordion span {
    font-size: 1.05rem !important;
    padding: 5px 10px !important;
    color: #475569 !important;                /* <--- Applicato #475569 */
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
}

/* 2. NOMI DEI FILE E RELATIVI RADIO BUTTON */
#album_col label span,
#album_col .label-wrap span,
#album_col input[type="radio"] + span {
    font-size: 1.05rem !important;
    color: #475569 !important;                /* <--- Applicato #475569 */
    font-weight: 500 !important;
}

#album_col .wrap {
    flex-direction: column !important;
    gap: 4px !important;
}


/* Aggiunge una linea orizzontale sotto l'header */
#app_header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 28px;
}

/* Definisce il blocco nome file */
#nome_file,
#nome_file > *,
#nome_file .block,
#nome_file .form,
#nome_file .wrap,
#nome_file .wrap-inner,
#nome_file .container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#nome_file {
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 2px 12px !important;
}
#nome_file input, #nome_file textarea {
    text-align: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #1e2d4a !important;
    letter-spacing: 0.01em !important;
    padding: 6px 0 !important;
}

/* Rimuove la box bianca aggiunta automaticamente da gradio sotto l'immagine */
#img_orig .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Definisce l'immagine originale: altezza massima 70% della finestra, proporzioni mantenute */
#img_orig img {
    width: 100% !important;
    height: auto !important;
    max-height: 70vh !important;
    display: block !important;
    object-fit: contain !important;
    border-radius: 10px !important;
}
#img_orig .image-container,
#img_orig [data-testid="image"] {
    height: auto !important;
    max-height: none !important;
}

/* Rimuove la box bianca aggiunta automaticamente da gradio sotto la maschera */
#mask_viewer {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Definisce la sezione tag e ocr */
/* Azzera il box che Gradio aggiunge al Group */
#ocr_block, #tags_block,
#ocr_block > .block, #tags_block > .block,
#ocr_block.block, #tags_block.block,
#ocr_block > div, #tags_block > div {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* Separatore orizzontale tra immagine e sezione */
#ocr_block, #tags_block {
    border-top: 1px solid #e2e8f0 !important;
    margin-top: 12px !important;
    padding-top: 12px !important;
}

/* Definisce il tab selector */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 16px !important;
}
.tabs > .tab-nav > button {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    transition: color 0.15s ease !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.tabs > .tab-nav > button.selected {
    color: #4361ee !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #4361ee !important;
    background: transparent !important;
}

/* Definisce il pulsante Salva */
button.primary,
button[data-testid="primary"],
.gradio-button.primary {
    background: #4361ee !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(67,97,238,0.28) !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
    font-size: 1.1rem !important;
}
button.primary:hover {
    opacity: 0.88 !important;
    box-shadow: 0 4px 18px rgba(67,97,238,0.38) !important;
}

/* Definisce il pulsante Annulla */
button.secondary,
button[data-testid="secondary"],
.gradio-button.secondary {
    background: #f1f5f9 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
    font-size: 1.1rem !important;
}
button.secondary:hover {
    background: #e2e8f0 !important;
}

/* ── Definisce il pusante Salva e procedi */
#btn_procedi > button {
    background: #1e2d4a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(15,23,42,0.22) !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
    font-size: 1.05rem !important;
}
#btn_procedi > button:hover {
    opacity: 0.85 !important;
    box-shadow: 0 6px 26px rgba(15,23,42,0.30) !important;
}

/* Definisce i tag */
.multiselect span.token {
    background: #eff6ff !important;
    color: #3b5bdb !important;
    border-radius: 6px !important;
    border: 1px solid #bfdbfe !important;
    font-size: 0.95rem !important;
}

/* Nasconde il pulsante "svuota tutto" (×) e la freccia dropdown (▼) nella sezione tag */
#dd_tags button { display: none !important; }
/* Ripristina i pulsanti × individuali dentro ogni token */
#dd_tags .token button { display: inline-flex !important; }

/* Definisce i messaggi di stato */
#ocr_block .prose p,
#tags_block .prose p {
    font-size: 1.05rem !important;
    margin: 8px 0 0 !important;
    color: #475569 !important;
}

/* Avviso modifiche non salvate — popup centrato al centro della pagina */
/* NOTA: Gradio duplica l'elem_id su due div annidati (wrapper esterno + contenuto interno).
   La regola di base deve quindi restare "invisibile" e con overflow visibile, altrimenti il
   wrapper esterno collassa attorno al proprio padding e ritaglia il popup vero e proprio. */
#dialog_unsaved {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000 !important;
    width: 90% !important;
    max-width: 360px !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}
/* Il div interno (quello con i contenuti reali) riceve l'aspetto grafico del popup */
#dialog_unsaved #dialog_unsaved {
    padding: 22px 24px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    /* la seconda ombra, con spread enorme, simula lo sfondo scuro semi-trasparente dietro al popup */
    box-shadow: 0 20px 50px -10px rgba(15,23,42,0.35), 0 0 0 9999px rgba(15,23,42,0.45) !important;
}
#dialog_unsaved > .block,
#dialog_unsaved .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#dialog_msg .prose p, #dialog_msg p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1e2d4a !important;
    margin: 0 0 18px !important;
    line-height: 1.5 !important;
    text-align: center !important;
}
#dialog_unsaved .row {
    justify-content: center !important;
}
"""


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------INTERFACCIA------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

try:                                                                         # crea un tema ed evita un crash se la versione di gradio non supporta quel tema
    _theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate")
except Exception:
    _theme = None

_theme_kwarg = {"theme": _theme} if _theme else {}

with gr.Blocks(title="Revisione immagini", css=custom_css, **_theme_kwarg) as demo:                                                  # definisce il contenitore di tutta l'app; viene passato il css personalizzato
    gr.HTML('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">')       # Carica il font Inter direttamente nell'html della pagina

    initial_reviewed = get_reviewed_images()        # definisce il set dei file già revisionati

    # Variabili di stato invisibili all'utente, ma accessibili da tutti i callback
    state_user         = gr.State("")                # nome dell'utente loggato
    state_img_path     = gr.State("")                # path dell'immagine corrente
    state_orig_txt     = gr.State("")                # target "Annulla" (versione precedente o output modello)
    state_orig_tags    = gr.State([])                # idem per i tag
    state_reviewed     = gr.State(initial_reviewed)  # set delle immagini già revisionate
    state_loaded_txt   = gr.State("")                # testo come era al caricamento (per rilevare modifiche)
    state_loaded_tags  = gr.State([])                # tag come erano al caricamento
    state_pending_path = gr.State("")                # path su cui l'utente vuole navigare (pending dialog)

    # Schermata d login
    with gr.Group(elem_id="login_card") as grp_login:
        gr.HTML('<h1 style="font-family:\'Inter\',sans-serif;color:#1e2d4a;font-size:1.9rem;font-weight:700;margin:0 0 6px;letter-spacing:-0.01em;">Progetto Caproni</h1><p style="font-family:\'Inter\',sans-serif;color:#64748b;margin:0 0 28px;font-size:1.05rem;">Inserisci i tuoi dati per accedere alla revisione.</p>')
        with gr.Row():
            inp_nome  = gr.Textbox(label="Nome e Cognome *", placeholder="Es. Mario Rossi")
            inp_ruolo = gr.Textbox(label="Ruolo (opzionale)", placeholder="Es. Ricercatore")
        btn_login     = gr.Button("Accedi →", variant="primary")
        msg_login_err = gr.Markdown(visible=False)

    # Interfaccia principale
    with gr.Group(visible=False) as grp_app:
        with gr.Row(elem_id="app_header"):
            lbl_utente = gr.Markdown("")

        with gr.Row():

            # Colonna 1 — albero album
            with gr.Column(scale=3, min_width=220, elem_id="album_col"):
                gr.Markdown("### Album")
                radios = []
                accordions = []
                for aname in album_names_sorted:
                    with gr.Accordion(aname, open=False) as acc:
                        init_choices = build_radio_choices(initial_reviewed, aname)
                        r = gr.Radio(choices=init_choices, label="Seleziona un'immagine", interactive=True)
                        radios.append(r)
                    accordions.append(acc)

            # Colonna principale — nome file, analizza, tab revisione
            with gr.Column(scale=8):

                # Avviso modifiche non salvate (visibile solo se necessario)
                with gr.Group(visible=False, elem_id="dialog_unsaved") as dialog_unsaved:
                    lbl_dialog_msg = gr.Markdown("", elem_id="dialog_msg")
                    with gr.Row():
                        btn_dialog_salva   = gr.Button("💾 Salva e procedi", variant="primary")
                        btn_dialog_annulla = gr.Button("✏️ Continua a modificare")

                # Nome file corrente
                txt_nome = gr.Textbox(value="Seleziona un'immagine dall'elenco di sinistra", label="", show_label=False, interactive=False, elem_id="nome_file", container=False)

                gr.HTML('<div style="height: 12px;"></div>')

                # Tab 1: immagine originale + editor testo OCR
                # Tab 2: immagine con maschere + editor tag
                with gr.Tabs(selected=0) as tabs_main:

                    with gr.TabItem("📄  Testo OCR", id=0):
                        img_orig = gr.Image(
                            value=None,
                            type="filepath",
                            label="Immagine originale",
                            interactive=False,
                            elem_id="img_orig",
                            container=False,
                        )
                        with gr.Group(elem_id="ocr_block"):
                            gr.HTML('<p style="font-family:\'Inter\',sans-serif;color:#64748b;font-size:0.82rem;font-weight:700;margin:0 0 12px;letter-spacing:0.10em;text-transform:uppercase;">Testo OCR <span style="font-size:0.78rem;font-weight:400;letter-spacing:0;text-transform:none;color:#94a3b8;">— Aggiungi e/o modifica il testo</span></p>')
                            txt_ocr = gr.Textbox(
                                label="", show_label=False, lines=4, interactive=True,
                                placeholder="",
                            )
                            with gr.Row():
                                btn_salva_ocr   = gr.Button("💾 Salva testo", variant="primary", scale=1)
                                btn_annulla_ocr = gr.Button("↩ Annulla", scale=1)
                            lbl_stato_ocr = gr.Markdown("")

                    with gr.TabItem("🏷️  Tag e maschere", id=1):
                        mask_viewer = gr.HTML("", elem_id="mask_viewer")
                        with gr.Group(elem_id="tags_block"):
                            gr.HTML('<p style="font-family:\'Inter\',sans-serif;color:#64748b;font-size:0.82rem;font-weight:700;margin:0 0 12px;letter-spacing:0.10em;text-transform:uppercase;">Tag <span style="font-size:0.78rem;font-weight:400;letter-spacing:0;text-transform:none;color:#94a3b8;">— Digita un tag e premi Invio per aggiungerlo; clicca × per rimuoverlo.</span></p>')
                            dd_tags = gr.Dropdown(
                                choices=[], value=[],
                                multiselect=True,
                                allow_custom_value=True,
                                label="", show_label=False,
                                interactive=True,
                                elem_id="dd_tags",
                            )
                            with gr.Row():
                                btn_salva_tags   = gr.Button("💾 Salva tag e procedi →", variant="primary", scale=1)
                                btn_annulla_tags = gr.Button("↩ Annulla", scale=1)
                            lbl_stato_tags = gr.Markdown("")



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------CALLBACKS--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------


    # Se il nome è vuoto dà errore, altrimenti salva il nome utente in state_user
    # .then aggiorna lbl_utente con il nome dell'utente loggato
    def do_login(nome, ruolo):
        if not nome.strip():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                gr.update(visible=True, value="⚠️ Il campo *Nome e Cognome* è obbligatorio."),
            )
        user = nome.strip() + (f" ({ruolo.strip()})" if ruolo.strip() else "")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            user,
            gr.update(visible=False, value=""),
        )

    _login_kwargs = dict(
        fn=do_login,
        inputs=[inp_nome, inp_ruolo],
        outputs=[grp_login, grp_app, state_user, msg_login_err],
    )
    _then_kwargs = dict(
        fn=lambda u: f"👤 **{u}**",
        inputs=[state_user],
        outputs=[lbl_utente],
    )

    btn_login.click(**_login_kwargs).then(**_then_kwargs)
    inp_nome.submit(**_login_kwargs).then(**_then_kwargs)
    inp_ruolo.submit(**_login_kwargs).then(**_then_kwargs)

    # Per ogni radio: carica l'immagine, ma se ci sono modifiche non salvate mostra il dialog
    _RADIO_OUTPUTS = (
        [img_orig, mask_viewer, txt_nome, dd_tags, txt_ocr,
         state_img_path, state_orig_txt, state_orig_tags,
         lbl_stato_ocr, lbl_stato_tags, tabs_main]
        + radios + accordions
        + [dialog_unsaved, lbl_dialog_msg, state_pending_path,
           state_loaded_txt, state_loaded_tags]
    )

    for i, r in enumerate(radios):
        def make_fn(radio_idx):
            def fn(path, curr_txt, curr_tags, loaded_txt, loaded_tags, curr_img_path):
                n_acc = len(accordions)
                n_rad = len(radios)
                no_op = [gr.update()] * len(_RADIO_OUTPUTS)

                if path is None or path == curr_img_path:
                    return no_op

                # Controlla modifiche non salvate
                txt_mod  = (curr_txt  or "").strip() != (loaded_txt  or "").strip()
                tags_mod = sorted(curr_tags or []) != sorted(loaded_tags or [])
                if curr_img_path and (txt_mod or tags_mod):
                    msg = "Salvare le modifiche effettuate?"
                    # Rimetti il radio sul valore corrente e mostra il dialog.
                    # curr_img_path appartiene all'album dell'immagine corrente, non necessariamente
                    # a quello del radio appena cliccato (radio_idx): se l'utente ha cambiato album,
                    # curr_img_path non è tra le scelte del radio cliccato e va deselezionato (None),
                    # altrimenti Gradio rifiuta il valore perché non presente in "choices".
                    curr_album = find_album_for_image(curr_img_path)
                    curr_album_idx = album_names_sorted.index(curr_album) if curr_album in album_names_sorted else None
                    radio_upds = []
                    for j in range(n_rad):
                        if j == curr_album_idx:
                            radio_upds.append(gr.update(value=curr_img_path))
                        elif j == radio_idx:
                            radio_upds.append(gr.update(value=None))
                        else:
                            radio_upds.append(gr.update())
                    return (
                        [gr.update()] * 11
                        + radio_upds
                        + [gr.update()] * n_acc
                        + [gr.update(visible=True), msg, path, gr.update(), gr.update()]
                    )

                # Nessuna modifica — naviga normalmente
                _mask_orig, nome, txt, tags, has_masks, has_text = load_image_data(path)
                annulla_txt, annulla_tags = load_annulla_target(Path(path).stem)
                overlays_b64 = get_overlay_b64(Path(path).stem, path)
                html = make_mask_viewer_html(path, overlays_b64)
                album = find_album_for_image(path)
                ocr_msg  = "" if has_text  else "⚠️ Non è stato identificato alcun testo"
                tags_msg = "" if has_masks else "⚠️ Non sono stati identificati alcun tag o maschere"
                base = [
                    path, html, nome,
                    gr.update(choices=tags, value=tags),
                    txt, path, annulla_txt, annulla_tags, ocr_msg, tags_msg,
                    gr.update(selected=0),
                ]
                radio_upds = [
                    gr.update() if j == radio_idx else None
                    for j in range(n_rad)
                ]
                acc_upds = [gr.update(open=(aname == album)) for aname in album_names_sorted]
                return (
                    base + radio_upds + acc_upds
                    + [gr.update(visible=False), "", "", txt, tags]
                )
            return fn

        r.change(
            fn=make_fn(i),
            inputs=[r, txt_ocr, dd_tags, state_loaded_txt, state_loaded_tags, state_img_path],
            outputs=_RADIO_OUTPUTS,
        )

    # Helper: aggiorna tutte le choices dei radio con i ✓
    def all_radio_updates(reviewed_set: set) -> list:
        _updates = []
        for aname in album_names_sorted:
            _updates.append(gr.update(choices=build_radio_choices(reviewed_set, aname)))
        return _updates

    # Passa alla tab dei tag senza salvare — il salvataggio avviene solo in do_salva_tags
    def do_salva_ocr(img_path):
        if not img_path:
            return "⚠️ Analizza prima un'immagine.", gr.update()
        return "", gr.update(selected=1)

    btn_salva_ocr.click(
        fn=do_salva_ocr,
        inputs=[state_img_path],
        outputs=[lbl_stato_ocr, tabs_main],
    )

    btn_annulla_ocr.click(fn=lambda o: o, inputs=[state_orig_txt], outputs=[txt_ocr])

    def _load_and_navigate(next_path, new_rev):
        """Carica next_path e costruisce il blocco di aggiornamenti comuni."""
        _mask_orig, nome, next_txt, next_tags, has_masks, has_text = load_image_data(next_path)
        annulla_txt, annulla_tags = load_annulla_target(Path(next_path).stem)
        next_name = Path(next_path).stem
        overlays_b64 = get_overlay_b64(next_name, next_path)
        html = make_mask_viewer_html(next_path, overlays_b64)
        next_album = find_album_for_image(next_path)
        ocr_msg  = "" if has_text  else "⚠️ Non è stato identificato alcun testo"
        tags_msg = "" if has_masks else "⚠️ Non è stata identificata nessun tag e maschera"
        radio_upds = []
        for aname in album_names_sorted:
            choices = build_radio_choices(new_rev, aname)
            value   = next_path if aname == next_album else None
            radio_upds.append(gr.update(choices=choices, value=value))
        acc_upds = [gr.update(open=(aname == next_album)) for aname in album_names_sorted]
        return (
            next_path, html, nome, next_txt,
            gr.update(choices=next_tags, value=next_tags),
            next_path, annulla_txt, annulla_tags,
            ocr_msg, tags_msg, new_rev, gr.update(selected=0),
            next_txt, next_tags,
        ), tuple(radio_upds), tuple(acc_upds)

    # Salva tag e procedi: salva, aggiorna ✓ nei radio e carica l'immagine successiva
    def do_salva_tags(tags, img_path, txt, user, reviewed):
        if not img_path:
            return (
                (gr.update(),) * 9
                + ("⚠️ Analizza prima un'immagine.",)
                + (reviewed, gr.update(), gr.update(), gr.update())
                + (gr.update(),) * len(radios)
                + (gr.update(),) * len(accordions)
            )
        name = Path(img_path).stem
        save_revision(name, img_path, find_mask_path(name), txt, tags or [], user)
        new_rev = set(reviewed) | {name}
        all_paths = [str(p) for p in lista_path_img]
        try:
            idx = all_paths.index(img_path)
        except ValueError:
            idx = -1
        next_path = all_paths[(idx + 1) % len(all_paths)]
        base, radio_upds, acc_upds = _load_and_navigate(next_path, new_rev)
        return base + radio_upds + acc_upds

    btn_salva_tags.click(
        fn=do_salva_tags,
        inputs=[dd_tags, state_img_path, txt_ocr, state_user, state_reviewed],
        outputs=[
            img_orig, mask_viewer, txt_nome, txt_ocr, dd_tags,
            state_img_path, state_orig_txt, state_orig_tags,
            lbl_stato_ocr, lbl_stato_tags,
            state_reviewed, tabs_main,
            state_loaded_txt, state_loaded_tags,
        ] + radios + accordions,
    )

    # Dialog "Rimani qui" — nasconde il dialog, nessuna navigazione
    btn_dialog_annulla.click(
        fn=lambda: (gr.update(visible=False), ""),
        inputs=[],
        outputs=[dialog_unsaved, state_pending_path],
    )

    # Dialog "Salva e vai" — salva l'immagine corrente e naviga al path in attesa
    def do_dialog_salva(tags, img_path, txt, user, reviewed, pending_path):
        if not img_path or not pending_path:
            return (gr.update(visible=False), "") + (gr.update(),) * (
                14 + len(radios) + len(accordions)
            )
        name = Path(img_path).stem
        save_revision(name, img_path, find_mask_path(name), txt, tags or [], user)
        new_rev = set(reviewed) | {name}
        base, radio_upds, acc_upds = _load_and_navigate(pending_path, new_rev)
        return (gr.update(visible=False), "") + base + radio_upds + acc_upds

    btn_dialog_salva.click(
        fn=do_dialog_salva,
        inputs=[dd_tags, state_img_path, txt_ocr, state_user, state_reviewed, state_pending_path],
        outputs=[
            dialog_unsaved, state_pending_path,
            img_orig, mask_viewer, txt_nome, txt_ocr, dd_tags,
            state_img_path, state_orig_txt, state_orig_tags,
            lbl_stato_ocr, lbl_stato_tags,
            state_reviewed, tabs_main,
            state_loaded_txt, state_loaded_tags,
        ] + radios + accordions,
    )

    btn_annulla_tags.click(
        fn=lambda orig: gr.update(choices=orig, value=orig),
        inputs=[state_orig_tags],
        outputs=[dd_tags],
    )



demo.launch(inbrowser=True, allowed_paths=[str(path_album), str(path_maschere_singole)])
