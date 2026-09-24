from pathlib import Path
import random
import pandas as pd
import csv
import os

def random_texts():

    Album_Trento_JPG_split = Path(r"/home/mwaldner/workspace/outputs/Album Trento JPG_split")

    lista_file_scelti = []
   
    for album in Album_Trento_JPG_split.iterdir():
        
        lista = []
        
        for txt in album.iterdir():
            txt_path = txt.as_posix()
            txt_name = txt.name
            txt_path_name = [txt_path, txt_name]
            lista += [txt_path_name]

        file_scelti = random.sample(lista, 10)
        lista_file_scelti.append(file_scelti)
    
    return lista_file_scelti


def csv_writer():

    dati = [
        ["file_name", "output", "expected_output", "check"]
    ]
    
    path = "/home/mwaldner/workspace/source/scripts/check_output_text2.csv"

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(dati)
            print("file csv creato")
            return path
    else:
        print("Il file csv esiste già")



def csv_compiler(lista_file_scelti, path_csv):

    with open(path_csv, "a", newline="", encoding="utf-8") as file_csv:
    
        for album in lista_file_scelti:
            
            for coppia in album:
                path_file_txt = coppia[0]
                file_name = str(coppia[1])

                with open(path_file_txt, "r", encoding="utf-8") as file_txt:
                    testo_estratto = file_txt.read()
                
                testo_pulito = testo_estratto.replace("\n", " ")

                scrittore = csv.writer(file_csv, delimiter=",")
                scrittore.writerow([file_name, testo_pulito])


path_csv = csv_writer()
if path_csv == None:
    print("Il file è già esistente, non viene sovrascritto")
else:
    lista_file_scelti = random_texts()
    print(csv_compiler(lista_file_scelti, path_csv))
    print("File csv scritto")
