import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import numpy as np
from pathlib import Path
from testfile import path_testfile, return_tag_from_image
from adjustText import adjust_text


# imposta il device
device = "cuda"

# carica il modello
model = build_sam3_image_model()
model.to(device)
processor = Sam3Processor(model, confidence_threshold = 0.5) # questo secondo argomento serve per modificare il livello di confidenza


# la funzione prende in input il path di un'immagine e restituisce le corrispondenti box, maschere e livelli di confidenza
def extract_boxes_masks(path):
    path = Path(path) 
    image = Image.open(path)
    tags = return_tag_from_image(path)
    image_name = path.stem
    width, height = image.size

    all_masks = []
    all_boxes = []
    all_scores = []
    all_tags = []

    if device == "cuda":
        # autocast converte i pesi del modello e i pixel dell'immagine dal formato a 32 bit a quello a 16 bit
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(image) # trasforma l'immagine in una matrice (embedding)
            
            for tag in tags:
                
                print(tag)
                if hasattr(processor, "reset_all_prompts"):
                    processor.reset_all_prompts(inference_state)
                
                output = processor.set_text_prompt(     
                    state = inference_state, 
                    prompt = tag, 
                    # box_threshold = 0.35,
                    # text_threshold = 0.30 # contiene le coordinate delle bounding box, le maschere e i punteggi di confidenza
                )

                num_detected = output["boxes"].shape[0]
                if num_detected > 0:
                    all_masks.append(output["masks"].float())
                    all_boxes.append(output["boxes"].float())
                    all_scores.append(output["scores"].float())
                    all_tags.extend([tag] * num_detected)

    if len(all_boxes) == 0:
        print(f"Nessun oggetto rilevato per i tag: {tags}")
        return image_name, torch.empty(0), torch.empty(0), torch.empty(0), []


    masks = torch.cat(all_masks, dim=0)
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)


    print(f"Rilevate {len(boxes)} maschere grezze con successo!")

    # questo step serve per mantenere solo le box migliori
    iou_threshold = 0.4

    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    keep_indices_list = keep_indices.cpu().tolist()

    boxes = boxes[keep_indices]
    masks = masks[keep_indices]
    scores = scores[keep_indices]
    tags_finali = [all_tags[idx] for idx in keep_indices_list]

    print(f"Dopo NMS: {len(boxes)} box rimaste.")

    # restituisce, separatamente in tre differenti variabili, maschere, boxes e livelli di confidenza
    return image_name, masks, boxes, scores, tags_finali


def save_predictions(image_name, masks, boxes, scores, tags, output_dir):
    image_output_dir = Path(output_dir) / image_name
    masks_dir = image_output_dir / "masks"
    boxes_dir = image_output_dir / "boxes"
    
    masks_dir.mkdir(exist_ok=True, parents=True)
    boxes_dir.mkdir(exist_ok=True, parents=True)
    
    boxes_data = []

    for i, (box, mask, score, tag) in enumerate(zip(boxes, masks, scores, tags)):
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np)
        mask_filename = f"{image_name}_mask_{i}_{tag}.png"
        mask_img.save(masks_dir / mask_filename)
    
        x1, y1, x2, y2 = box.cpu().numpy().tolist()
        boxes_data.append({
            "id": i,
            "box": [x1, y1, x2, y2],
            "score": float(score),
            "tag": tag,
            "mask_filename": mask_filename
        })
    
    with open(boxes_dir / f"{image_name}_boxes.json", "w") as f:
        json.dump(boxes_data, f, indent=4)
    print(f"Predizioni salvate nella cartella dedicata: {image_output_dir}")




def aggregate_masks_by_tag(masks, tags):
    """
    Raggruppa le maschere per tag e le unisce con un'operazione OR logica:
    un pixel appartiene all'oggetto aggregato se appartiene ad almeno una
    delle istanze con quel tag (es. tutte le "person" diventano una sola maschera).
    Restituisce un dizionario {tag: maschera_aggregata (tensor booleano)}.
    """
    aggregated = {}
    for mask, tag in zip(masks, tags):
        mask_bool = mask.squeeze(0).bool()
        if tag not in aggregated:
            aggregated[tag] = mask_bool.clone()
        else:
            aggregated[tag] = aggregated[tag] | mask_bool
    return aggregated


def save_aggregated_masks(image_name, aggregated_masks, output_dir):
    """
    Salva una sola maschera per tag (unione di tutte le istanze con quel tag),
    in una sottocartella separata da quella delle maschere per-istanza.
    Nome file: {image_name}_{tag}.png — compatibile con load_masks_for_image
    del Gradio app, che estrae il tag direttamente dal nome del file.
    """
    agg_dir = Path(output_dir) / image_name / "masks_aggregated"
    agg_dir.mkdir(exist_ok=True, parents=True)

    for tag, mask in aggregated_masks.items():
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np)
        safe_tag = tag.replace("/", "_").replace("\\", "_")  # evita caratteri non validi nel path
        mask_img.save(agg_dir / f"{image_name}_{safe_tag}.png")
    print(f"Maschere aggregate per tag salvate in: {agg_dir}")




def image_with_boxes_masks(image_path, boxes_masks_dir, output_dir):
    image_path = Path(image_path)
    image_name = image_path.stem
    
    data_dir = Path(boxes_masks_dir) / image_name

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    json_path = data_dir / "boxes" / f"{image_name}_boxes.json"
    if not json_path.exists():
        print(f"Nessun dato trovato per {image_name} in {json_path}")
        return

    with open(json_path, "r") as f:
        boxes_data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_np)

    texts = []
    rects = []

    for item in boxes_data:
        i = item["id"]
        box = item["box"]
        score = item["score"]
        tag_label = item.get("tag", "N/A")

        mask_filename = item.get("mask_filename", f"{image_name}_mask_{i}.png")
        mask_path = data_dir / "masks" / mask_filename
        if not mask_path.exists():
            continue
        
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img) / 255.0

        color = np.random.random(3)

        overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 4))
        overlay[..., :3] = color
        overlay[..., 3] = mask_np * 0.25  # Applica trasparenza
        
        ax.imshow(overlay)

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=0.4,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)
        rects.append(rect)

        t = ax.text(
            x1,
            y1 - 10,
            f"{tag_label}: {score:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(
                facecolor=color,
                alpha=0.1, # trasparenza rettangoli etichette
                edgecolor="none",
                boxstyle="square,pad=0.3"
            )
        )
        texts.append(t)

    if texts:
        adjust_text(
            texts, 
            objects=rects, 
            arrowprops=dict(arrowstyle="-", color="white", lw=0.5),
            expand=(1.2, 1.4)  # Spazio di tolleranza attorno al testo (orizzontale, verticale)
        )

    plt.axis("off")

    verification_out_dir = Path(output_dir)
    verification_out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = verification_out_dir / f"{image_name}_verification.png"
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Immagine di verifica salvata in: {output_path}")


# # SCRIPT PER ESEGUIRE SOLO UN'IMMAGINE
# if __name__ == "__main__":
    
#     lista_test = path_testfile()
#     if not lista_test:
#         print("La lista dei file di test è vuota")
#         exit()
        
#     test_image_path = lista_test[8] # seleziona solo un'immagine per la verifica
#     print(f"Avvio test su singola immagine: {test_image_path}")
    
#     # cartelle di output
#     base_output_dir = "/data/caproni/maddalena_waldner/workspace/outputs/prova_sam3_2409/predizioni"
#     verification_dir = "/data/caproni/maddalena_waldner/workspace/outputs/prova_sam3_2409/verifica_immagini"

#     # estrazione con SAM3 + NMS
#     img_name, masks, boxes, scores, tags = extract_boxes_masks(test_image_path)
    
#     if boxes.shape[0] > 0:
#         # salvataggio: crea in automatico base_output_dir/img_name/masks e base_output_dir/img_name/boxes
#         save_predictions(img_name, masks, boxes, scores, tags, base_output_dir)
#         print("Fase di salvataggio completata con successo")
        
#         # genera l'immagine di verifica leggendo dalla nuova struttura
#         image_with_boxes_masks(
#             image_path=test_image_path,
#             boxes_masks_dir=base_output_dir,
#             output_dir=verification_dir
#         )
#     else:
#         print(f"nessuna box rilevata per l'immagine {img_name}: il salvataggio viene saltato")
    
#     print("Test completato")



# # SCRIPT PER ESEGUIRE TUTTE LE IMMAGINI ALL'INTERNO DELLA LISTA DELLE 100 IMMAGINI DI PROVA
# if __name__ == "__main__":
    
#     lista_test = path_testfile()
#     if not lista_test:
#         print("La lista dei file di test è vuota")
#         exit()

#     for path in lista_test:

#         test_image_path = path
    
#         # cartelle di output
#         base_output_dir = "/data/caproni/maddalena_waldner/workspace/outputs/prova_sam3_2409/predizioni"
#         verification_dir = "/data/caproni/maddalena_waldner/workspace/outputs/prova_sam3_2409/verifica_immagini"

#         # estrazione con SAM3 + NMS
#         img_name, masks, boxes, scores, tags = extract_boxes_masks(test_image_path)
        
#         # se non ci sono tag per un'immagine il ciclo continua
#         if boxes.numel() == 0:
#             print(f"Nessun tag trovato per {img_name}")
#             continue

#         # salvataggio: crea in automatico base_output_dir/img_name/masks e base_output_dir/img_name/boxes
#         save_predictions(img_name, masks, boxes, scores, tags, base_output_dir)
#         aggregated = aggregate_masks_by_tag(masks, tags)
#         save_aggregated_masks(img_name, aggregated, base_output_dir)
#         print("Fase di salvataggio completata con successo")
        
#         # genera l'immagine di verifica leggendo dalla nuova struttura
#         image_with_boxes_masks(
#             image_path=test_image_path,
#             boxes_masks_dir=base_output_dir,
#             output_dir=verification_dir
#         )

#     print("Test completato")


# SCRIPT PER ESEGUIRE TUTTE LE IMMAGINI
if __name__ == "__main__":
    
    path_album_img = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
    lista_path_img = []
    for album in path_album_img.iterdir():
        for img in album.iterdir():
            lista_path_img.append(img)
    if not lista_path_img:
        print("La lista dei file di test è vuota")
        exit()

    for path in lista_path_img:

        test_image_path = path
    
        # cartelle di output
        base_output_dir = "/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni"
        verification_dir = "/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/verifica_immagini"

        # estrazione con SAM3 + NMS
        img_name, masks, boxes, scores, tags = extract_boxes_masks(test_image_path)
        
        # se non ci sono tag per un'immagine il ciclo continua
        if boxes.numel() == 0:
            print(f"Nessun tag trovato per {img_name}")
            continue

        # salvataggio: crea in automatico base_output_dir/img_name/masks e base_output_dir/img_name/boxes
        save_predictions(img_name, masks, boxes, scores, tags, base_output_dir)
        aggregated = aggregate_masks_by_tag(masks, tags)
        save_aggregated_masks(img_name, aggregated, base_output_dir)
        print("Fase di salvataggio completata con successo")
        
        # genera l'immagine di verifica leggendo dalla nuova struttura
        image_with_boxes_masks(
            image_path=test_image_path,
            boxes_masks_dir=base_output_dir,
            output_dir=verification_dir
        )

    print("Test completato")