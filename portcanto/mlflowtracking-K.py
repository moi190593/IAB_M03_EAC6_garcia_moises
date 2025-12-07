""" @ IOC - Moises Garcia - 2025 - CE IABD """

import sys
import logging
import mlflow

from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Carrega el dataset
    df = load_dataset('./data/ciclistes.csv')
    df_clean = clean(df)
    true_labels, df_features = extract_true_labels(df_clean)

    # Crea l'experiment a MLflow
    experiment_name = "K sklearn ciclistes"
    mlflow.set_experiment(experiment_name)

    for k in range(2, 9):  # K de 2 a 8
        with mlflow.start_run(run_name=f"K={k}"):
            # Entrena el model
            model = clustering_kmeans(df_features, n_clusters=k)
            pred_labels = model.labels_

            # Calcula les mètriques
            homo = homogeneity_score(true_labels, pred_labels)
            comp = completeness_score(true_labels, pred_labels)
            vmeas = v_measure_score(true_labels, pred_labels)

            # Log dels paràmetres i mètriques
            mlflow.log_param("K", k)
            mlflow.log_metric("homogeneity", homo)
            mlflow.log_metric("completeness", comp)
            mlflow.log_metric("v_measure", vmeas)

            logging.info(f"K={k} | homogeneity={homo:.3f} | completeness={comp:.3f} | v_measure={vmeas:.3f}")

    print("s'han generat els runs")