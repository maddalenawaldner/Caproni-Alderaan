import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Modello più grande = istruzioni seguite meglio (serve ~16 GB di GPU).
# Se la GPU è piccola, rimetti "meta-llama/Llama-3.2-3B-Instruct".
model_id = "Qwen/Qwen2.5-7B-Instruct"

print(f"Caricamento del modello {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    return_full_text=False,
    do_sample=False
)

tag_estratti = ['photo', 'white', 'black', 'person', 'man', 'image', 'stand', 'plane', 'boat', 'sea', 'water', 'sky', 'biplane', 'crowd', 'building', 'fly', 'blanket', 'uniform', 'officer', 'propeller', 'crowded', 'collect', 'vessel', 'cloudy', 'pilot', 'army', 'picture', 'city', 'woman', 'formation', 'soldier', 'wear', 'book', 'walk', 'floor', 'album', 'pose', 'field', 'lake', 'troop', 'seaplane', 'calm', 'sailor', 'cloud', 'journal', 'air field', 'room', 'float', 'open', 'flag', 'notebook', 'land', 'brown', 'sit', 'city view', 'bird', 'hardback book', 'table', 'pillar', 'dress', 'waterway', 'poster', 'parade', 'group photo', 'line', 'crew', 'car', 'blue', 'tie', 'dirt field', 'writing', 'road', 'dock', 'hat', 'equipment', 'stool', 'rail', 'city skyline', 'portrait', 'job', 'white house', 'shake', 'canoe', 'ferry', 'shoreline', 'ladder', 'pole', 'paper', 'tarmac', 'travel', 'plywood', 'battleship', 'city street', 'sun hat', 'doorway', 'horizon', 'hut', 'plain', 'couple', 'deck', 'family photo', 'workshop', 'sign', 'flock', 'tree', 'hangar', 'warehouse', 'fighter jet', 'color', 'ruler', 'copy', 'harbor', 'attach', 'factory', 'tool', 'coach', 'cane', 'hand', 'wave', 'boardwalk', 'text', 'crane', 'fog', 'rowboat', 'grass', 'ink', 'telescope', 'mountain', 'urban', 'lighthouse', 'cockpit', 'sheet', 'guard', 'window', 'cloth', 'green', 'march', 'pattern', 'salute', 'warship', 'letter', 'cart', 'horse', 'wagon', 'document', 'town', 'catch', 'beach', 'balustrade', 'machine', 'aircraft carrier', 'hall', 'postcard', 'gun', 'landscape', 'sand', 'house', 'ride', 'grassy', 'square', 'motorcycle', 'snow', 'palace', 'river', 'train track', 'assembly', 'write', 'stormy', 'park', 'motorboat', 'folder', 'rifle', 'suit', 'child', 'cowboy hat', 'smoke', 'beam', 'wing', 'frame', 'bulletin board', 'footprint', 'scaffold', 'choir', 'church', 'town square', 'commander', 'monument', 'stair', 'chair', 'wood', 'draw', 'load', 'gathering', 'train car', 'palm tree', 'dinning table', 'food', 'sail', 'author', 'skier', 'stone', 'night view', 'ramp', 'family', 'fabric', 'assembly line', 'flag pole', 'ceiling', 'ice', 'stone building', 'pavement', 'plaza', 'city square', 'spire', 'street corner', 'coast', 'notepad', 'carpenter', 'workbench', 'rectangle', 'animal', 'pull', 'hillside', 'signature', 'lamp', 'map', 'tent', 'plank', 'shed', 'sailboat', 'night', 'fishing pole', 'alley', 'play', 'ledge', 'baseball team', 'baseball bat', 'student', 'cruise ship', 'talk', 'steam', 'sew', 'string', 'strip', 'tripod', 'board', 'clock', 'cross', 'drawing', 'football game', 'obelisk', 'archway', 'train', 'sculpture', 'statue', 'crack', 'corridor', 'rock formation', 'scenery', 'trolley', 'plaque', 'debris', 'dirt road', 'house exterior', 'office', 'paddle', 'donkey', 'note', 'tarp', 'tablecloth', 'canteen', 'canopy', 'bend', 'log', 'shovel', 'book cover', 'power line', 'camp', 'bat', 'crop', 'fishing boat', 'raft', 'tire', 'drummer', 'instrument', 'horseback', 'mud', 'baseball uniform', 'baseball glove', 'classroom', 'material', 'twine', 'ski pole', 'cage', 'key', 'produce', 'menu', 'iceberg', 'mast', 'motorcyclist', 'football field', 'stadium', 'sing', 'ski', 'arch', 'hay', 'skyscraper', 'headdress', 'passenger train', 'railroad', 'area', 'peak', 'tower', 'nest', 'ballroom', 'street sign', 'duck', 'measuring tape', 'convertible', 'street scene', 'mark', 'balcony', 'stare', 'cruiser', 'barge', 'drive', 'beret', 'fence', 'grave', 'post', 'airplanewindow', 'flood', 'armchair', 'umbrella', 'island', 'boy', 'destruction', 'row', 'rural', 'worker', 'laboratory', 'bicycle', 'barn', 'kayak', 'oar', 'wood wall', 'spin', 'red', 'envelope', 'screw', 'carpet', 'crate', 'beard', 'dark', 'overpass', 'seawall', 'amusement park', 'lumber', 'apron', 'waiter', 'fiction', 'pen', 'laugh', 'smile', 'screen door', 'harvest', 'moonlight', 'tube', 'lift', 'anchor', 'baseball player', 'class', 'school', 'boxer', 'muscle', 'shirtless', 'tugboat', 'dye', 'fiber', 'reel', 'wool', 'suspenders', 'bookshelf', 'flight', 'drag', 'push', 'wash', 'wedding party', 'grassland', 'tank', 'baby carriage', 'clay', 'pottery', 'glider', 'fruit', 'melon', 'squash', 'antique', 'fruit stand', 'snowy', 'rocket', 'bookstore', 'golfer', 'design', 'motorbike', 'landing', 'well', 'amphitheater', 'case', 'shadow', 'bird nest', 'spider web', 'web', 'telegraph pole', 'swim', 'storm cloud', 'record player', 'sink', 'hallway', 'track', 'fish', 'porch', 'lean', 'window sill', 'tape', 'bay', 'out', 'coffin', 'jeep', 'fleet', 'throne', 'arm', 'doctor', 'hospital', 'hospital bed', 'hospital room', 'lab coat', 'nurse', 'patient', 'staff', 'cliff', 'terrain', 'valley', 'graveyard','spectator', 'mule', 'plow', 'submarine', 'cocktail dress', 'cannon', 'curtain', 'biker', 'sword', 'cardboard', 'needle', 'pin', 'tail', 'bomb', 'shore', 'pillow', 'raceway', 'hanger', 'remain', 'saw', 'skeleton', 'hammer', 'loom', 'wall', 'furniture', 'mill', 'computer', 'brick building', 'narrow', 'wheel', 'repair', 'chandelier', 'phone box', 'bow tie', 'antenna', 'water tower', 'cake', 'flight attendant', 'art', 'bus stop', 'park bench', 'mat', 'Eiffel tower', 'parachute', 'wedding dress', 'bride', 'wedding photo']

# Rimuove i duplicati mantenendo l'ordine
tag_unici = list(dict.fromkeys(tag_estratti))

SYSTEM = """You classify tags for image segmentation with SAM3.
The images are scans of historical photo albums (early 20th century, aviation).

KEEP a tag if it names a concrete, visible thing that occupies a region of the image
and could be outlined with a mask: objects, people, animals, vehicles, buildings,
and background regions such as sky, sea, grass.

REMOVE a tag if it is:
- a color (white, blue, brown)
- an adjective or atmosphere (crowded, calm, cloudy, dark, rural)
- a verb or action (stand, fly, walk, wear, shake)
- an abstract concept or category (job, travel, fiction, design, area)
- a description of the whole picture, not of a part of it
  (photo, image, picture, portrait, group photo, landscape, scenery, city view)
- part of the album itself rather than the photo content
  (album, book, hardback book, notebook, blanket, shadow)

Answer ONLY with a JSON object mapping EVERY input tag to "keep" or "remove".
No explanations."""

# Esempio che mostra al modello il formato esatto
ESEMPIO_INPUT = ["plane", "white", "fly", "propeller", "photo", "crowded", "sky"]
ESEMPIO_OUTPUT = {"plane": "keep", "white": "remove", "fly": "remove",
                  "propeller": "keep", "photo": "remove", "crowded": "remove", "sky": "keep"}


def classifica(batch):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(ESEMPIO_INPUT)},
        {"role": "assistant", "content": json.dumps(ESEMPIO_OUTPUT)},
        {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
    ]
    output = pipe(messages)[0]["generated_text"]
    if isinstance(output, list):  # alcune versioni restituiscono la chat intera
        output = output[-1]["content"]

    # Estrae il JSON dalla risposta, anche se il modello aggiunge testo attorno
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        print("Risposta non valida:", output[:200])
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        print("JSON non valido:", match.group()[:200])
        return {}


BATCH_SIZE = 25
risultati = {}

for i in range(0, len(tag_unici), BATCH_SIZE):
    batch = tag_unici[i:i + BATCH_SIZE]
    print(f"Batch {i // BATCH_SIZE + 1}: {len(batch)} tag")
    risposta = classifica(batch)
    for tag in batch:
        risultati[tag] = risposta.get(tag, "da_verificare")  # tag saltati dal modello

da_tenere = [t for t, v in risultati.items() if v == "keep"]
da_rimuovere = [t for t, v in risultati.items() if v == "remove"]
da_verificare = [t for t, v in risultati.items() if v not in ("keep", "remove")]

print(f"\nDa tenere: {len(da_tenere)}")
print(f"Da rimuovere: {len(da_rimuovere)}")
print(f"Da verificare a mano: {len(da_verificare)}")

with open("classificazione_tag.json", "w", encoding="utf-8") as f:
    json.dump({"da_tenere": da_tenere,
               "da_rimuovere": da_rimuovere,
               "da_verificare": da_verificare}, f, ensure_ascii=False, indent=2)