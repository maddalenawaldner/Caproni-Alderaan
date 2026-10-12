import json
from pathlib import Path
from pymongo import MongoClient

# --- Settings ---
JSONL_FOLDER = Path("/data/caproni/maddalena_waldner/workspace/outputs/gradio/revisioni")
DB_NAME = "caproni"
COLLECTION_NAME = "revisioni"
# ----------------

client = MongoClient("mongodb://localhost:27017")
client.admin.command("ping")
collection = client[DB_NAME][COLLECTION_NAME]

files = sorted(JSONL_FOLDER.glob("*.jsonl"))
print(f"Found {len(files)} JSONL files")

total = 0
errors = 0

for file in files:
    batch = []
    with open(file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                errors += 1
                print(f"{file.name}, line {line_number}: invalid JSON, skipped ({e})")
                continue
            doc["file_origine"] = file.name  # remember which file the record came from
            batch.append(doc)

    if batch:
        collection.insert_many(batch)
        total += len(batch)
    print(f"{file.name}: {len(batch)} records imported")

print(f"\nTotal imported: {total}, lines skipped: {errors}")
print(f"Documents now in '{COLLECTION_NAME}': {collection.count_documents({})}")

client.close()