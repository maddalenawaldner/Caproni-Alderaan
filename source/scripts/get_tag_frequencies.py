from pathlib import Path


def get_tag_frequencies():
    
    outputs_ram = Path("/home/mwaldner/workspace/outputs/outputs_ram")
    dict_tag_frequencies = {}

    for album in outputs_ram.iterdir():
        for txt in album.iterdir():
            with open(txt, "r") as file:
                lista = file.read()

                lista_tag = lista.split(",")

                for tag in lista_tag:
                    tag_pulito = tag.strip()

                    if tag_pulito in dict_tag_frequencies:
                        dict_tag_frequencies[tag_pulito] += 1
                    else:
                        dict_tag_frequencies[tag_pulito] = 1
                
    dict_tag_frequencies_ordered = dict(sorted(dict_tag_frequencies.items(), key=lambda item: item[1], reverse=True))
    
    return dict_tag_frequencies_ordered




def filter_tag():

    outputs_ram = Path("/home/mwaldner/workspace/outputs/outputs_ram")
    outputs_ram_clean = Path("/home/mwaldner/workspace/outputs/outputs_ram_clean")

    for album in outputs_ram.iterdir():
        
        album_dest = outputs_ram_clean / (album.name + "_clean")
        album_dest.mkdir(parents=True, exist_ok=True)

        for txt in album.iterdir():
            
            with open(txt, "r") as file:
                lista = file.read()
                lista_tag = lista.split(",")
                
                lista_tag_pulita = []
                
                for tag in lista_tag:
                    tag_pulito = tag.strip()
                    if tag_pulito not in ["photo", "white", "black", "picture", "image", "blanket", "album", "book", "hardback book", "notebook", "shadow"]:
                        lista_tag_pulita.append(tag_pulito)
                    else:
                        continue

                txt_dest = album_dest / txt.name

                with open(txt_dest, "w", encoding="utf-8") as outFile:
                    outFile.write(", ".join(lista_tag_pulita))
                


dict_tag_frequencies = get_tag_frequencies()
print(dict_tag_frequencies)

