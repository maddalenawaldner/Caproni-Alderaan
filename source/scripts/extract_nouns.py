import json
from pathlib import Path
from vllm import LLM, SamplingParams

# carica il modello
llm = LLM(
    model="Qwen/Qwen3-8B",
    max_model_len=8192,             # lunghezza massima prompt + risposta
    gpu_memory_utilization=0.85,    # quota di memoria GPU che vLLM può occupare
)

# temperatura 0 = risposte deterministiche, adatto all'estrazione
sampling = SamplingParams(temperature=0.0, max_tokens=1024)

ISTRUZIONI = (
    "Estrai tutti i sostantivi dal testo fornito includendo solo nomi concreti ed escludendo nomi propri."
    "Riportali in minuscolo, senza articoli e senza ripetizioni. "
    "Rispondi SOLO con una lista di stringhe, ad esempio: [\"casa\", \"aereo\"]. "
    "Se non ci sono sostantivi rispondi []."
)

def build_messages(testo: str) -> list:
    return [
        {"role": "system", "content": ISTRUZIONI},
        {"role": "user", "content": testo},
    ]

# converte la risposta del modello in lista; se non è JSON valido restituisce una lista vuota
def parse_output(risposta: str) -> list:
    risposta = risposta.strip()
    inizio, fine = risposta.find("["), risposta.rfind("]")
    if inizio == -1 or fine == -1:
        return []
    try:
        return json.loads(risposta[inizio:fine + 1])
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":

    path_album_testi = Path("/data/caproni/maddalena_waldner/workspace/outputs/Album Trento JPG_split")
    path_output      = Path("/data/caproni/maddalena_waldner/workspace/outputs/sostantivi_qwen_txt")

    # raccoglie tutti i testi e i relativi path
    testi = []
    paths = []
    for album in sorted(path_album_testi.iterdir()):
        if not album.is_dir():
            continue
        for txt in sorted(album.iterdir()):
            if txt.suffix.lower() != ".txt":
                continue
            testo = txt.read_text(encoding="utf-8").strip()
            if not testo:                                  # salta i testi OCR vuoti
                continue
            print(testo)
            testi.append(testo)
            paths.append(txt)

    print(f"Testi da elaborare: {len(testi)}")

    conversazioni = [build_messages(t) for t in testi]

    # tutti i testi in un'unica chiamata: vLLM li elabora in parallelo
    outputs = llm.chat(
        conversazioni,
        sampling,
        chat_template_kwargs={"enable_thinking": False},   # disattiva il "ragionamento" di Qwen3
    )

    # salva un file TXT per immagine, mantenendo la struttura a cartelle degli album
    for txt_path, out in zip(paths, outputs):
        sostantivi = parse_output(out.outputs[0].text)

        out_dir = path_output / txt_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{txt_path.stem}.txt").write_text(
            ", ".join(sostantivi), encoding="utf-8"
        )

    print(f"Risultati salvati in: {path_output}")
