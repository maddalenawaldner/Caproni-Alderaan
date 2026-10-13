from pathlib import Path
import pandas as pd
import csv
import os
from testfile import path_testfile, return_tag_from_image

def csv_writer():

    dati = [
        ["file_name", "tags"]
    ]
    
    path = "/home/mwaldner/workspace/source/scripts/check_output_boxes_masks.csv"

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(dati)
            print("file csv creato")
            return path
    else:
        print("Il file csv esiste già")



def csv_compiler(lista_path_testfile, path_csv):

    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        scrittore = csv.writer(f, delimiter=",")

        for path in lista_path_testfile:
            path_image = Path(path)
            lista_tag = return_tag_from_image(path_image)
            riga_da_scrivere = [path_image.name] + [lista_tag]
            scrittore.writerow(riga_da_scrivere)


if __name__ == "__main__":
    lista_path_testfile = path_testfile()
    path_csv = csv_writer()

    if path_csv == None:
        print("Il file è già esistente, non viene sovrascritto")
    else:
        check_boxes_masks_csv = csv_compiler(lista_path_testfile, path_csv)
        print(check_boxes_masks_csv)
        print("File csv scritto")
