from pathlib import Path
import shutil
import re

documenti = Path("data/Album-Trento JPG")
output_base = Path("data/Album Trento JPG_split")
output_base.mkdir(exist_ok=True)




def images_filter():

    # ciclo su ogni cartella di immagini
    for cartella in documenti.iterdir():

        print("Trovata cartella:", cartella.name)
        if cartella.is_dir():

            cartella_dest = output_base / (cartella.name + "_split")
            cartella_dest.mkdir(exist_ok=True)

            cartella_jpg = cartella / "JPEG"

            # all'interno di ogni cartella, ciclo sui file.
            for jpg in cartella_jpg.iterdir():
                print("     Trovato file:", jpg.name)

                if jpg.is_file() and jpg.suffix == ".jpg":

                    match1 = re.search(r"^A.*_(\d+)$", jpg.stem)
                    match2 = re.search(r"^B", jpg.stem)

                    if match1: 

                        numero = int(match1.group(1))

                        if numero % 2 == 1:
                            shutil.copy(jpg, cartella_dest / jpg.name)

                    elif match2:
                        shutil.copy(jpg, cartella_dest / jpg.name)
        else:
            print("non è una directory")

        
    return None


images_filter()