# metrics_server.py
from prometheus_client import Gauge, start_http_server
import time
import json
import os

# Initialiser les gauges
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model', ['model_name'])
auc_gauge = Gauge('model_auc', 'AUC-ROC of the model', ['model_name'])
recall_gauge = Gauge('model_recall', 'Recall of the model', ['model_name'])
precision_gauge = Gauge('model_precision', 'Precision of the model', ['model_name'])
fpr_gauge = Gauge('model_false_positive_rate', 'False Positive Rate of the model', ['model_name'])
training_time_gauge = Gauge('model_training_time_seconds', 'Training time in seconds', ['model_name'])

# Chemin vers le fichier où les métriques sont enregistrées par Reentrainement.py
METRICS_FILE = "/usr/local/airflow/include/scripts/metrics.json"

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        return metrics
    return {}

# Démarrer le serveur HTTP et mettre à jour les métriques en boucle
start_http_server(8002)  # Utilise le port 8002 pour éviter conflit avec spark-worker
print("Serveur HTTP démarré sur le port 8002. Attente infinie...")

while True:
    metrics = load_metrics()
    for model_name, values in metrics.items():
        accuracy_gauge.labels(model_name=model_name).set(float(values.get('accuracy', 0)))
        auc_gauge.labels(model_name=model_name).set(float(values.get('auc', 0)))
        recall_gauge.labels(model_name=model_name).set(float(values.get('recall', 0)))
        precision_gauge.labels(model_name=model_name).set(float(values.get('precision', 0)))
        fpr_gauge.labels(model_name=model_name).set(float(values.get('false_positive_rate', 0)))
        training_time_gauge.labels(model_name=model_name).set(float(values.get('training_time', 0)))  # Utilise 'training_time' au lieu de 'training_time_seconds'
    time.sleep(10)  # Met à jour toutes les 10 secondes