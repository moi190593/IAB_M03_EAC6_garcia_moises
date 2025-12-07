# Port del Cantó – EAC6 – IOC CE IABD

Aquest projecte forma part de l’EAC6 del mòdul 3. Es genera un conjunt de dades sintètiques de ciclistes i posteriorment s’aplica un procés de clustering amb KMeans per identificar diferents perfils de ciclistes segons el seu rendiment en pujada i baixada.

L’objectiu és practicar:
- generació de dades sintètiques
- clustering amb scikit-learn
- mètriques d’avaluació
- visualització
- informes
- MLflow
- tests unitaris
- documentació

El projecte utilitza Python 3, numpy, pandas, scikit-learn, matplotlib, seaborn i MLflow.

## Scripts principals

generardataset.py:
Genera dades sintètiques dels ciclistes mitjançant distribucions normals i les desa al fitxer “data/ciclistes.csv”. S’hi generen quatre perfils de ciclistes: BEBB, BEMB, MEBB i MEMB.

clustersciclistes.py:
Carrega el dataset, realitza Exploratory Data Analysis, neteja de dades, extracció de labels reals, entrenament de KMeans, càlcul de mètriques (homogeneity, completeness, v-measure), visualització, generació d’informes, guardat del model i associació cluster–tipus.

mlflowtracking-K.py:
Crea un experiment a MLflow i executa clustering variant el valor de K entre 2 i 8, registrant homogeneity, completeness i v-measure.

## Carpetes principals del projecte

data/            dataset generat
img/             imatges (pairplot, clusters)
informes/        informes per tipus de ciclista
model/           models guardats i diccionaris
tests/           tests unitaris

## Execució

Instal·lació de dependències:
    pip install -r requirements.txt

Generació dataset:
    python generardataset.py

Entrenament i clustering:
    python clustersciclistes.py

## MLflow

Executar primer:
    mlflow ui

I després:
    python mlflowtracking-K.py

A MLflow es poden consultar els resultats comparatius per valors de K.

## Tests

Per executar els tests unitaris:
    pytest -v

## Autor
Moises Garcia Pedemonte — IOC 2025/26

## Llicència
El projecte utilitza la llicència MIT (vegeu LICENSE.txt).
