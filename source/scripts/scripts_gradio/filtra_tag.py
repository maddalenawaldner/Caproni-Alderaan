from pathlib import Path


def filtra_tag(name_img):

    path_tags = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")
    path_tag_sopravvissuti = Path("/data/caproni/maddalena_waldner/workspace/outputs/outputs_sam3_tutti/predizioni")

    for album in path_tag_sopravvissuti.iterdir():
        if album.name == name_img:
            print(album.name)
            for cartella in album.iterdir():
                if cartella.name == "masks_aggregated":

                    lista_tag_sopravvissuti = []
                    
                    for file in cartella.iterdir():
                        stem = file.stem
                        tag = stem.split("_")[-1]
                        lista_tag_sopravvissuti.append(tag)
                        print(tag)
    
    return(lista_tag_sopravvissuti)




print(filtra_tag("A-1314_00035"))


# -------------------------

