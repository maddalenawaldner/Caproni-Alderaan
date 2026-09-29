import json
from pathlib import Path
from vllm import LLM, SamplingParams

# Configurazione modello
llm = LLM(
    model="Qwen/Qwen3-8B",
    max_model_len=8192,             # lunghezza massima prompt + risposta
    gpu_memory_utilization=0.85,    # quota di memoria GPU che vLLM può occupare
)

# temperatura 0 = risposte deterministiche
sampling = SamplingParams(temperature=0.0, max_tokens=2048)

ISTRUZIONI = (
    "Traduci dall'inglese all'italiano le seguenti parole.\n"
    "Restituisci le parole tradotte in minuscolo, senza articoli e senza ripetizioni.\n"
    "Mantieni la stessa struttura del testo di input. Non aggiungere spiegazioni, numeri o testo introduttivo."
    "Se non c'è nessuna parola non tradurre niente, lascia vuoto."
)

def build_messages(testo: str) -> list:
    return [
        {"role": "system", "content": ISTRUZIONI},
        {"role": "user", "content": testo},
    ]


if __name__ == "__main__":

    # Percorsi base
    dir_input  = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
    dir_output = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean_ita")

    testi = []
    paths = []

    # Scorre le sottocartelle e raccoglie tutti i file .txt
    for album in sorted(dir_input.iterdir()):
        for txt_path in sorted(album.iterdir()):
            if txt_path.suffix.lower() != ".txt":
                continue
            
            testo = txt_path.read_text(encoding="utf-8").strip()
            
            testi.append(testo)
            paths.append(txt_path)

    print(f"File .txt trovati da elaborare: {len(testi)}")

    if testi:
        # Prepara il batch di conversazioni
        conversazioni = [build_messages(t) for t in testi]

        # Generazione in parallelo con vLLM
        outputs = llm.chat(
            conversazioni,
            sampling,
            chat_template_kwargs={"enable_thinking": False},  # disattiva il "ragionamento" di Qwen3
        )

        # Salvataggio ricreando la struttura delle cartelle
        for txt_path, out in zip(paths, outputs):
            risposta_tradotta = out.outputs[0].text.strip()

            # Calcola il percorso relativo per mantenere la stessa struttura di cartelle
            percorso_relativo = txt_path.relative_to(dir_input)
            out_file_path = dir_output / percorso_relativo

            # Crea la sottocartella di destinazione se non esiste
            out_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Scrive il file tradotto
            out_file_path.write_text(risposta_tradotta, encoding="utf-8")

        print(f"Tutti i file tradotti sono stati salvati in: {dir_output}")
    else:
        print("Nessun file valido da elaborare.")