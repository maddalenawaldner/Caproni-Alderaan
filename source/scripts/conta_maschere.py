from pathlib import Path

verifica_immagini = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/verifica_immagini")

i = 0
for file in verifica_immagini.iterdir():
    i += 1

print(i)

album_trento = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")

j = 0
for album in album_trento.iterdir():
    for file in album.iterdir():
        j += 1

print(j)

outputs_ram = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")

z = 0 
for album in outputs_ram.iterdir():
    for file in album.iterdir():
        if file.stat().st_size == 0:
            z += 1

print(z)

# -------------------------------------------------------------------------------------------------------------------------------------

for album in album_trento.iterdir():
    for file in album.iterdir():
        for album in verifica_immagini.iterdir():
            for file in album.iterdir():
                