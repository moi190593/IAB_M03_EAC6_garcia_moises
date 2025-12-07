"""
@ IOC - CE IABD
"""
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
        path -- dataset

    Returns: dataframe
    """
    df = pd.read_csv(path)
    return df

def eda(df):
    """
    Exploratory Data Analysis del dataframe

    arguments:
        df -- dataframe

    Returns: None
    """
    logging.info("Primeres files:\n%s", df.head())
    logging.info("Descripció estadística:\n%s", df.describe())
    logging.info("Tipus de dades:\n%s", df.dtypes)

def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
        df -- dataframe

    Returns: dataframe
    """
    return df.drop(['id', 'tt'], axis=1)

def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
        df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    labels = df['tipus'].values
    df_features = df.drop('tipus', axis=1)
    return labels, df_features

def visualitzar_pairplot(df):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    arguments:
        df -- dataframe

    Returns: None
    """
    os.makedirs('img', exist_ok=True)
    sns.pairplot(df)
    plt.savefig('img/pairplot.png')
    plt.close()

def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model

    arguments:
        data -- les dades: tp i tb

    Returns: model (objecte KMeans)
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    os.makedirs('img', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(data['tp'], data['tb'], c=labels, cmap='viridis', s=40)
    plt.xlabel('Temps pujada')
    plt.ylabel('Temps baixada')
    plt.title('Clusters de ciclistes')
    plt.savefig('img/clusters.png')
    plt.close()

def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    dicc = {'tp': 0, 'tb': 1}
    logging.info('Centres:')
    for idx, tipus_item in enumerate(tipus):
        logging.info(
            '%d:\t(tp: %.1f\ttb: %.1f)',
            idx,
            model.cluster_centers_[idx][dicc['tp']],
            model.cluster_centers_[idx][dicc['tb']]
        )
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1
    suma_max = 0
    suma_min = 50000
    for j, center in enumerate(model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j
    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})
    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)
    if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]
    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})
    return tipus

def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    os.makedirs('informes', exist_ok=True)
    for tipus_item in tipus:
        label = tipus_item['label']
        name = tipus_item['name']
        subset = df[df['label'] == label]
        with open(f'informes/{name}.txt', 'w', encoding='utf-8') as informe:
            informe.write(f"Ciclistes del tipus {name} (label {label}):\n")
            informe.write(subset.to_string(index=False))

def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    df_nous_pred = pd.DataFrame(dades, columns=['id', 'tp', 'tb', 'tt'])
    df_nous_clean = df_nous_pred.drop(['id', 'tt'], axis=1)
    pred_labels = model.predict(df_nous_clean)
    df_nous_pred['label'] = pred_labels
    return df_nous_pred, pred_labels

# ----------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Iniciant el procés de clustering de ciclistes...")

    os.makedirs('model', exist_ok=True)
    os.makedirs('informes', exist_ok=True)
    os.makedirs('img', exist_ok=True)

    path_dataset = './data/ciclistes.csv'

    # 1. Carrega dataset
    logging.info("Carregant el dataset...")
    dataframe = load_dataset(path_dataset)

    # 2. EDA
    logging.info("Realitzant EDA...")
    eda(dataframe)

    # 3. Neteja
    logging.info("Netejant el dataset...")
    df_cleaned = clean(dataframe)

    # 4. Extracció de labels i eliminació de columna tipus
    logging.info("Extraient etiquetes reals i eliminant columna tipus...")
    true_labels, df_features = extract_true_labels(df_cleaned)

    # 5. Visualització pairplot
    logging.info("Generant img/pairplot.png...")
    visualitzar_pairplot(df_features)

    # 6. Clustering KMeans
    logging.info("Entrenant el model KMeans...")
    clustering_model = clustering_kmeans(df_features, n_clusters=4)

    # 7. Guardar model
    logging.info("Guardant el model a model/clustering_model.pkl...")
    with open('model/clustering_model.pkl', 'wb') as model_file:
        pickle.dump(clustering_model, model_file)

    # 8. Scores
    pred_labels = clustering_model.labels_
    scores = {
        'homogeneity': homogeneity_score(true_labels, pred_labels),
        'completeness': completeness_score(true_labels, pred_labels),
        'v_measure': v_measure_score(true_labels, pred_labels)
    }
    with open('model/scores.pkl', 'wb') as scores_file:
        pickle.dump(scores, scores_file)
    logging.info("Scores: %s", scores)

    # 9. Visualitzar clusters
    logging.info("Generant img/clusters.png...")
    df_features['label'] = pred_labels
    visualitzar_clusters(df_features, pred_labels)

    # 10. Associar clusters a patrons
    logging.info("Associant clusters als patrons de comportament...")
    tipus_list = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus_list = associar_clusters_patrons(tipus_list, clustering_model)

    # 11. Guardar tipus_dict
    logging.info("Guardant la correspondència tipus-label a model/tipus_dict.pkl...")
    with open('model/tipus_dict.pkl', 'wb') as tipus_file:
        pickle.dump(tipus_list, tipus_file)

    # 12. Generar informes
    logging.info("Generant informes a la carpeta informes/...")
    generar_informes(df_features, tipus_list)

    # 13. Classificació de nous valors
    logging.info("Classificant nous ciclistes...")
    nous_ciclistes = [
        [500, 3230, 1430, 4670], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550]  # MEMB
    ]
    df_nous_pred, pred_nous = nova_prediccio(nous_ciclistes, clustering_model)
    for i, pred_label in enumerate(pred_nous):
        tipus_match = [tt for tt in tipus_list if tt['label'] == pred_label]
        logging.info(
            'Nou ciclista id %s classificat com %s (label %s)',
            df_nous_pred.iloc[i]['id'], tipus_match[0]['name'], pred_label
        )

    logging.info("Procés completat correctament.")
