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
processor = Sam3Processor(model, confidence_threshold = 0.4) # questo secondo argomento serve per modificare il livello di confidenza


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
                    m = output["masks"].float()
                    b = output["boxes"].float()
                    s = output["scores"].float()

                    keep = torchvision.ops.nms(b, s, iou_threshold=0.4)
        
                    all_masks.append(m[keep])
                    all_boxes.append(b[keep])
                    all_scores.append(s[keep])
                    all_tags.extend([tag] * len(keep))


    if len(all_boxes) == 0:
        print(f"Nessun oggetto rilevato per i tag: {tags}")
        return image_name, torch.empty(0), torch.empty(0), torch.empty(0), []

    masks = torch.cat(all_masks, dim=0)
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)

    print(f"Rilevate {len(boxes)} maschere/box totali distribuite sui tag.")

    # restituisce, separatamente in tre differenti variabili, maschere, boxes e livelli di confidenza
    return image_name, masks, boxes, scores, all_tags


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
        mask_img.save(masks_dir / f"{image_name}_mask_{i}.png")
    
        x1, y1, x2, y2 = box.cpu().numpy().tolist()
        boxes_data.append({
            "id": i,
            "box": [x1, y1, x2, y2],
            "score": float(score),
            "tag": tag
        })
    
    with open(boxes_dir / f"{image_name}_boxes.json", "w") as f:
        json.dump(boxes_data, f, indent=4)
    print(f"Predizioni salvate nella cartella: {image_output_dir}")


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

        mask_path = data_dir / "masks" / f"{image_name}_mask_{i}.png"
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
        
#     test_image_path = "/home/mwaldner/workspace/data/Album Trento JPG_split/A-1315_split/A-1315_00021.jpg" # lista_test[3] # seleziona solo un'immagine per la verifica
#     print(f"Avvio test su singola immagine: {test_image_path}")
    
#     # cartelle di output
#     base_output_dir = "/home/mwaldner/workspace/outputs/prove_sam3_2/predizioni"
#     verification_dir = "/home/mwaldner/workspace/outputs/prove_sam3_2/verifica_immagini"

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



# SCRIPT PER ESEGUIRE TUTTE LE IMMAGINI ALL'INTERNO DELLA LISTA DELLE 100 IMMAGINI DI PROVA
if __name__ == "__main__":
    
    lista_test = path_testfile()
    if not lista_test:
        print("La lista dei file di test è vuota")
        exit()

    for path in lista_test:

        test_image_path = path
    
        # cartelle di output
        base_output_dir = "/home/mwaldner/workspace/outputs/prove_sam3_sinonimi/predizioni"
        verification_dir = "/home/mwaldner/workspace/outputs/prove_sam3_sinonimi/verifica_immagini"

        # estrazione con SAM3 + NMS
        img_name, masks, boxes, scores, tags = extract_boxes_masks(test_image_path)
        
        # se non ci sono tag per un'immagine il ciclo continua
        if boxes.numel() == 0:
            print(f"Nessun tag trovato per {img_name}")
            continue

        # salvataggio: crea in automatico base_output_dir/img_name/masks e base_output_dir/img_name/boxes
        save_predictions(img_name, masks, boxes, scores, tags, base_output_dir)
        print("Fase di salvataggio completata con successo")
        
        # genera l'immagine di verifica leggendo dalla nuova struttura
        image_with_boxes_masks(
            image_path=test_image_path,
            boxes_masks_dir=base_output_dir,
            output_dir=verification_dir
        )

    print("Test completato")