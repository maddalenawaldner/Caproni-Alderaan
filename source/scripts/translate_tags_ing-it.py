# import json
# from pathlib import Path
# from vllm import LLM, SamplingParams

# # Configurazione modello
# llm = LLM(
#     model="Qwen/Qwen3-8B",
#     max_model_len=8192,             # lunghezza massima prompt + risposta
#     gpu_memory_utilization=0.85,    # quota di memoria GPU che vLLM può occupare
# )

# # temperatura 0 = risposte deterministiche
# sampling = SamplingParams(temperature=0.0, max_tokens=2048)

# ISTRUZIONI = (
#     "Traduci dall'inglese all'italiano le seguenti parole.\n"
#     "Restituisci le parole tradotte in minuscolo, senza articoli e senza ripetizioni.\n"
#     "Mantieni la stessa struttura del testo di input. Non aggiungere spiegazioni, numeri o testo introduttivo."
#     "Se non c'è nessuna parola non tradurre niente, lascia vuoto."
# )

# def build_messages(testo: str) -> list:
#     return [
#         {"role": "system", "content": ISTRUZIONI},
#         {"role": "user", "content": testo},
#     ]


# if __name__ == "__main__":

#     # Percorsi base
#     dir_input  = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
#     dir_output = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_ita")

#     testi = []
#     paths = []

#     # Scorre le sottocartelle e raccoglie tutti i file .txt
#     for album in sorted(dir_input.iterdir()):
#         for txt_path in sorted(album.iterdir()):
#             if txt_path.suffix.lower() != ".txt":
#                 continue
            
#             testo = txt_path.read_text(encoding="utf-8").strip()
            
#             testi.append(testo)
#             paths.append(txt_path)

#     print(f"File .txt trovati da elaborare: {len(testi)}")

#     if testi:
#         # Prepara il batch di conversazioni
#         conversazioni = [build_messages(t) for t in testi]

#         # Generazione in parallelo con vLLM
#         outputs = llm.chat(
#             conversazioni,
#             sampling,
#             chat_template_kwargs={"enable_thinking": False},  # disattiva il "ragionamento" di Qwen3
#         )

#         # Salvataggio ricreando la struttura delle cartelle
#         for txt_path, out in zip(paths, outputs):
#             risposta_tradotta = out.outputs[0].text.strip()

#             # Calcola il percorso relativo per mantenere la stessa struttura di cartelle
#             percorso_relativo = txt_path.relative_to(dir_input)
#             out_file_path = dir_output / percorso_relativo

#             # Crea la sottocartella di destinazione se non esiste
#             out_file_path.parent.mkdir(parents=True, exist_ok=True)

#             # Scrive il file tradotto
#             out_file_path.write_text(risposta_tradotta, encoding="utf-8")

#         print(f"Tutti i file tradotti sono stati salvati in: {dir_output}")
#     else:
#         print("Nessun file valido da elaborare.")




import json
from pathlib import Path
from vllm import LLM, SamplingParams

dir_input       = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
path_dizionario = Path("/data/caproni/maddalena_waldner/workspace/outputs/tag_en_ita.json")

ISTRUZIONI = (
    "Traduci dall'inglese all'italiano la parola o espressione che ti viene data.\n"
    "Rispondi solo con la traduzione, in minuscolo, senza articoli, senza punteggiatura e senza spiegazioni."
)

def build_messages(tag: str) -> list:
    return [
        {"role": "system", "content": ISTRUZIONI},
        {"role": "user", "content": tag},
    ]


if __name__ == "__main__":

    # 1. Raccoglie tutti i tag inglesi unici
    tag_unici = set()
    for album in sorted(dir_input.iterdir()):
        for txt_path in sorted(album.glob("*.txt")):
            for t in txt_path.read_text(encoding="utf-8").split(","):
                t = t.strip()
                if t:
                    tag_unici.add(t)
    print(f"Tag unici trovati: {len(tag_unici)}")

    # 2. Riprende il dizionario esistente: i tag già tradotti (o corretti a mano) non vengono ritradotti
    dizionario = {}
    if path_dizionario.exists():
        dizionario = json.loads(path_dizionario.read_text(encoding="utf-8"))
    da_tradurre = sorted(t for t in tag_unici if t not in dizionario)
    print(f"Tag da tradurre: {len(da_tradurre)}")

    # 3. Traduce un tag per richiesta (tutte in parallelo con vLLM)
    if da_tradurre:
        llm = LLM(model="Qwen/Qwen3-8B", max_model_len=8192, gpu_memory_utilization=0.85)
        sampling = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = llm.chat(
            [build_messages(t) for t in da_tradurre],
            sampling,
            chat_template_kwargs={"enable_thinking": False},
        )
        for tag, out in zip(da_tradurre, outputs):
            dizionario[tag] = out.outputs[0].text.strip().strip(".").lower()

    # 4. Salva il dizionario {tag_inglese: tag_italiano}
    path_dizionario.write_text(
        json.dumps(dict(sorted(dizionario.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dizionario salvato in: {path_dizionario}")