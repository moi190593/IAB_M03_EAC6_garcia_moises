"""
Genera un dataset sintètic de ciclistes i el desa a data/ciclistes.csv
"""

import logging
import csv
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generar_dataset(num, ind, dicc):
    """
    Genera dades sintètiques de ciclistes.

    arguments:
        num -- nombre de ciclistes per tipus
        ind -- identificador inicial
        dicc -- llista de diccionaris amb paràmetres per cada tipus

    Returns: llista de llistes amb [id, tp, tb, tt, tipus]
    """
    data = []
    current_id = ind

    for tipo_dict in dicc:
        tipo_name = tipo_dict["name"]
        mu_p = tipo_dict["mu_p"]  # mitjana temps pujada
        mu_b = tipo_dict["mu_b"]  # mitjana temps baixada
        sigma = tipo_dict["sigma"]  # desviació estàndard

        for _ in range(num):
            tp = int(np.random.normal(mu_p, sigma))
            tb = int(np.random.normal(mu_b, sigma))
            tt = tp + tb
            data.append([current_id, tp, tb, tt, tipo_name])
            current_id += 1

    return data

if __name__ == "__main__":
    STR_CICLISTES = 'data/ciclistes.csv'

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240  # mitjana temps pujada bons escaladors
    MU_P_ME = 4268  # mitjana temps pujada mals escaladors
    MU_B_BB = 1440  # mitjana temps baixada bons baixadors
    MU_B_MB = 2160  # mitjana temps baixada mals baixadors
    SIGMA = 240     # 240 s = 4 min

    dicc = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    # Generar el dataset: 25 ciclistes per cada tipus (100 total)
    dataset = generar_dataset(25, 1, dicc)

    # Escriure a CSV
    with open(STR_CICLISTES, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tp', 'tb', 'tt', 'tipus'])  # capçalera
        writer.writerows(dataset)

    logging.info("s'ha generat data/ciclistes.csv")