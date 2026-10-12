import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Modello leggero ed efficiente (es. Gemma 2 2B o 9B istruito)
model_id = "google/gemma-2-2b-it"

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
    max_new_tokens=1000,
    return_full_text=False,
    do_sample=False
)

tag_estratti = ['photo', 'white', 'black', 'person', 'man', 'image', 'stand', 'plane', 'boat', 'sea', 'water', 'sky', 'biplane', 'crowd', 'building', 'fly', 'blanket', 'uniform', 'officer', 'propeller', 'crowded', 'collect', 'vessel', 'cloudy', 'pilot', 'army', 'picture', 'city', 'woman', 'formation', 'soldier', 'wear', 'book', 'walk', 'floor', 'album', 'pose', 'field', 'lake', 'troop', 'seaplane', 'calm', 'sailor', 'cloud', 'journal', 'air field', 'room', 'float', 'open', 'flag', 'notebook', 'land', 'brown', 'sit', 'city view', 'bird', 'hardback book', 'table', 'pillar', 'dress', 'waterway', 'poster', 'parade', 'group photo', 'line', 'crew', 'car', 'blue', 'tie', 'dirt field', 'writing', 'road', 'dock', 'hat', 'equipment', 'stool', 'rail', 'city skyline', 'portrait', 'job', 'white house', 'shake', 'canoe', 'ferry', 'shoreline', 'ladder', 'pole', 'paper', 'tarmac', 'travel', 'plywood', 'battleship', 'city street', 'sun hat', 'doorway', 'horizon', 'hut', 'plain', 'couple', 'deck', 'family photo', 'workshop', 'sign', 'flock', 'tree', 'hangar', 'warehouse', 'fighter jet', 'color', 'ruler', 'copy', 'harbor', 'attach', 'factory', 'tool', 'coach', 'cane', 'hand', 'wave', 'boardwalk', 'text', 'crane', 'fog', 'rowboat', 'grass', 'ink', 'telescope', 'mountain', 'urban', 'lighthouse', 'cockpit', 'sheet', 'guard', 'window', 'cloth', 'green', 'march', 'pattern', 'salute', 'warship', 'letter', 'cart', 'horse', 'wagon', 'document', 'town', 'catch', 'beach', 'balustrade', 'machine', 'aircraft carrier', 'hall', 'postcard', 'gun', 'landscape', 'sand', 'house', 'ride', 'grassy', 'square', 'motorcycle', 'snow', 'palace', 'river', 'train track', 'assembly', 'write', 'stormy', 'park', 'motorboat', 'folder', 'rifle', 'suit', 'child', 'cowboy hat', 'smoke', 'beam', 'wing', 'frame', 'bulletin board', 'footprint', 'scaffold', 'choir', 'church', 'town square', 'commander', 'monument', 'stair', 'chair', 'wood', 'draw', 'load', 'gathering', 'train car', 'palm tree', 'dinning table', 'food', 'sail', 'author', 'skier', 'stone', 'night view', 'ramp', 'family', 'fabric', 'assembly line', 'flag pole', 'ceiling', 'ice', 'stone building', 'pavement', 'plaza', 'city square', 'spire', 'street corner', 'coast', 'notepad', 'carpenter', 'workbench', 'rectangle', 'animal', 'pull', 'hillside', 'signature', 'lamp', 'map', 'tent', 'plank', 'shed', 'sailboat', 'night', 'fishing pole', 'alley', 'play', 'ledge', 'baseball team', 'baseball bat', 'student', 'cruise ship', 'talk', 'steam', 'sew', 'string', 'strip', 'tripod', 'board', 'clock', 'cross', 'drawing', 'football game', 'obelisk', 'archway', 'train', 'sculpture', 'statue', 'crack', 'corridor', 'rock formation', 'scenery', 'trolley', 'plaque', 'debris', 'dirt road', 'house exterior', 'office', 'paddle', 'donkey', 'note', 'tarp', 'tablecloth', 'canteen', 'canopy', 'bend', 'log', 'shovel', 'book cover', 'power line', 'camp', 'bat', 'crop', 'fishing boat', 'raft', 'tire', 'drummer', 'instrument', 'horseback', 'mud', 'baseball uniform', 'baseball glove', 'classroom', 'material', 'twine', 'ski pole', 'cage', 'key', 'produce', 'menu', 'iceberg', 'mast', 'motorcyclist', 'football field', 'stadium', 'sing', 'ski', 'arch', 'hay', 'skyscraper', 'headdress', 'passenger train', 'railroad', 'area', 'peak', 'tower', 'nest', 'ballroom', 'street sign', 'duck', 'measuring tape', 'convertible', 'street scene', 'mark', 'balcony', 'stare', 'cruiser', 'barge', 'drive', 'beret', 'fence', 'grave', 'post', 'airplanewindow', 'flood', 'armchair', 'umbrella', 'island', 'boy', 'destruction', 'row', 'rural', 'worker', 'laboratory', 'bicycle', 'barn', 'kayak', 'oar', 'wood wall', 'spin', 'red', 'envelope', 'screw', 'carpet', 'crate', 'beard', 'dark', 'overpass', 'seawall', 'amusement park', 'lumber', 'apron', 'waiter', 'fiction', 'pen', 'laugh', 'smile', 'screen door', 'harvest', 'moonlight', 'tube', 'lift', 'anchor', 'baseball player', 'class', 'school', 'boxer', 'muscle', 'shirtless', 'tugboat', 'dye', 'fiber', 'reel', 'wool', 'suspenders', 'bookshelf', 'flight', 'drag', 'push', 'wash', 'wedding party', 'grassland', 'tank', 'baby carriage', 'clay', 'pottery', 'glider', 'fruit', 'melon', 'squash', 'antique', 'fruit stand', 'snowy', 'rocket', 'bookstore', 'golfer', 'design', 'motorbike', 'landing', 'well', 'amphitheater', 'case', 'shadow', 'bird nest', 'spider web', 'web', 'telegraph pole', 'swim', 'storm cloud', 'record player', 'sink', 'hallway', 'track', 'fish', 'porch', 'lean', 'window sill', 'tape', 'bay', 'out', 'coffin', 'jeep', 'fleet', 'throne', 'arm', 'doctor', 'hospital', 'hospital bed', 'hospital room', 'lab coat', 'nurse', 'patient', 'staff', 'cliff', 'terrain', 'valley', 'graveyard','spectator', 'mule', 'plow', 'submarine', 'cocktail dress', 'cannon', 'curtain', 'biker', 'sword', 'cardboard', 'needle', 'pin', 'tail', 'bomb', 'shore', 'pillow', 'raceway', 'hanger', 'remain', 'saw', 'skeleton', 'hammer', 'loom', 'wall', 'furniture', 'mill', 'computer', 'brick building', 'narrow', 'wheel', 'repair', 'chandelier', 'phone box', 'bow tie', 'antenna', 'water tower', 'cake', 'flight attendant', 'art', 'bus stop', 'park bench', 'mat', 'Eiffel tower', 'parachute', 'wedding dress', 'bride', 'wedding photo']

# prompt = f"""
# You are an expert in object segmentation using SAM3.

# TASK:
# Analyze the following list of tags extracted from scanned photographs.
# Identify the tags that are not useful as text prompts for segmenting
# visual elements with SAM3.

# EXCLUDE:
# - Generic references to photographs, scans, and albums.
# - Colors and generic visual properties, such as white, black, and dark.
# - Shadows and other lighting-related terms.
# - Actions, verbs, and abstract concepts that do not identify
#   a specific, segmentable object.
# - Vague or redundant tags that do not help identify visual objects.

# KEEP:
# - Concrete objects, people, animals, vehicles, buildings,
#   structures, and meaningful scene elements.
# - Tags that could be useful to segment distinct regions,
#   even if they describe a scene rather than a physical object.

# IMPORTANT:
# - Only select tags that appear in the input list.
# - Do not invent or modify tags.
# - Do not remove a tag simply because it is infrequent.
# - Return ONLY a valid JSON list of strings.
# - Do not include explanations or markdown.

# INPUT TAGS:
# {json.dumps(tag_estratti, ensure_ascii=False)}
# """

prompt = f"""
TASK:
tag_estratti is the output of a model that extracted tags for the elements present in a scan of various photo albums. 
I need to use these extracted tags to segment the elements in the image with SAM3. 
However, the extracted tags dictionary contains several tags that are useless for segmentation (e.g., "photo", "white", "black", "picture", "image", "blanket", "album", "book", "hardback book", "notebook", "shadow").
Among the tags included in the dictionary I attached, are there any others that you think could be removed because they are useless for segmentation? Could you make a list of the tags present in the dictionary that are useless for segmentation and return it to me as a list?

INPUT TAGS:
{json.dumps(tag_estratti, ensure_ascii=False)}
"""


# Formattazione del prompt per i modelli chat (come Gemma-it)
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(formatted_prompt)
print("\n--- RISPOSTA DEL MODELLO ---")
print(outputs[0]["generated_text"])