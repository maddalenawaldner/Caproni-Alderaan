from pathlib import Path


def path_testfile():
    file_studio = []
    with open("/data/caproni/maddalena_waldner/workspace/source/scripts/check_output_text2.csv", "r") as f:
        next(f)
        for line in f:
            lista = line.strip().split(",")
            for i in range(len(lista)):
                if i == 0:
                    percorso_img = str(Path(lista[i]).with_suffix(".jpg"))
                    file_studio.append(percorso_img)

    
    Album_Trento_JPG_split = Path(r"/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split")
    lista_path = []
    for cartella in Album_Trento_JPG_split.iterdir():
        for elemento in cartella.iterdir():
            lista_path.append(elemento)
    
    lista_path_testfile = []
    for file in file_studio:
        for path_originale in lista_path:
            if Path(path_originale).name == file:
                lista_path_testfile.append(path_originale)
    
    return lista_path_testfile
 


def return_tag_from_image(path):

    outputs_ram_clean = Path(r"/data/caproni/maddalena_waldner/workspace/outputs/outputs_ram_clean")

    image_name = path.name.removesuffix(".jpg")

    for cartella in outputs_ram_clean.iterdir():
        for elemento in cartella.iterdir():
            elemento_name = elemento.name.removesuffix(".txt")
            if image_name == elemento_name:
                with open(elemento, "r") as f:
                    stringa_tags = f.read()
                    lista_tag = stringa_tags.split(",")
                    lista_tag_pulita = []
                    for tag in lista_tag:
                        tag_pulito = tag.strip()
                        lista_tag_pulita.append(tag_pulito)
                        
    return(lista_tag_pulita)


if __name__ == "__main__":
    path = Path("/data/caproni/maddalena_waldner/workspace/data/Album Trento JPG_split/A-1312_split/A-1312_00065.jpg")
    tag = return_tag_from_image(path)
    print(tag)

    lista_path_testfile = path_testfile()
    print(lista_path_testfile)





