import json
from pathlib import Path
from vllm import LLM, SamplingParams

path_dict_ita_eng = Path("/data/caproni/maddalena_waldner/workspace/outputs/tag_en_ita.json")
path_jsonl_tag_revisionati = Path("/data/caproni/maddalena_waldner/workspace/outputs/gradio/revisioni")
path_tag_revisionati = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")


# 1. Istruzioni per il modello
ISTRUZIONI = (
    "Traduci dall'italiano all'inglese il tag fornito."
    "Lascia inalterati i nomi propri."
    "Restituisci la traduzione in minuscolo, senza articoli."
    "Non aggiungere spiegazioni, virgolette o testo introduttivo."
    "Restituisci soltanto la traduzione."
)

# 2. Configurazione del modello (una sola volta)
llm = LLM(
    model="Qwen/Qwen3-8B",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
)

# 3. Parametri di generazione
sampling = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)



# funzione per tradurre utilizzando dizionario --> prende in input il path dei file jsonl revisionati e legge i tag in italiano; traduce utilizzando il dizionario i tag in inglese (se non ci sono nel dizionario li traduce con Qwen); modifica i file txt per rirunnare Sam3
def traduci_e_salva_tag(path):
        
    with open(path, "r", encoding="utf-8") as f:
        dizionario = json.load(f)
        dizionario_inverso = {valore: chiave for chiave, valore in dizionario.items()}
    
    for jsonl in path.iterdir():
        tags = []
        with jsonl.open("r", encoding="utf-8") as f:
            for riga in f:
                dato = json.loads(riga)
                if dato["versione"] == "corrente":
                    tags.extend(dato["tag"])
                    nome_file = dato["nome_file"]
                    print(nome_file)
        
        tags_tradotti = []
        tags_mancanti = []

        for tag in tags:
            if tag in dizionario_inverso:
                tags_tradotti.append(dizionario_inverso[tag])

            else:
                tags_mancanti.append(tag)
        
        if tags_mancanti:
            conversazioni = [
            [
                {"role": "system", "content": ISTRUZIONI},
                {"role": "user", "content": tag}
            ]
            
            for tag in tags_mancanti
        ]

            outputs = llm.chat(
                conversazioni,
                sampling,
                chat_template_kwargs={"enable_thinking": False}
            )

            traduzioni_qwen = [
                output.outputs[0].text.strip().lower()
                for output in outputs
            ]

            tags_tradotti.extend(traduzioni_qwen)


        for cartella in path_tag_revisionati.iterdir():
            for txt in cartella.iterdir():
                if txt.stem == nome_file:
                    print(txt.stem)
                    with open(txt, "w", encoding="utf-8") as f:
                        contenuto = ", ".join(tags_tradotti)
                        txt.write_text(contenuto, encoding="utf-8")






            



print(traduci_e_salva_tag(path_jsonl_tag_revisionati))



