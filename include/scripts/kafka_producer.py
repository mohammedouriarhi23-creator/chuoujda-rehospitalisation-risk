import json
import time
import pandas as pd
from kafka import KafkaProducer
import os

time.sleep(10)

# Chemin du fichier pour stocker l'état
STATE_FILE = './include/scripts/last_processed_index.json'
DURATION_SECONDS = 50

def read_last_index():
    """Lit l'index de la dernière ligne traitée depuis le fichier d'état."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f).get('last_index', 0)
        return 0  # Si le fichier n'existe pas, commencer à 0
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier d'état : {str(e)}")
        return 0

def save_last_index(index):
    """Sauvegarde l'index de la dernière ligne traitée."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({'last_index': index}, f)
        print(f"Sauvegarde de l'index {index} dans {STATE_FILE}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'index : {str(e)}")
        raise

try:
    # Initialiser le producteur Kafka
    producer = KafkaProducer(
        bootstrap_servers=os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5,
        acks='all'
    )
    print("Kafka producer initialisé avec succès")

    # Charger le dataset
    data_path = './include/data/streaming_data.csv'
    print(f"Chargement du dataset depuis {data_path}")
    data = pd.read_csv(data_path)
    topic = os.environ.get('KAFKA_TOPIC', 'hospital_fin67')
    print(f"Dataset chargé avec {len(data)} lignes. Envoi vers le topic : {topic}")

    # Lire l'index de départ
    start_index = read_last_index()
    print(f"Démarrage à partir de l'index {start_index}")

    # Suivre le temps de départ
    start_time = time.time()
    last_index_processed = start_index

    # Simuler l'envoi en temps réel pendant 2 minutes
    for index, row in data.iloc[start_index:].iterrows():
        # Vérifier si 2 minutes se sont écoulées
        if time.time() - start_time >= DURATION_SECONDS:
            print("2 minutes écoulées, arrêt de l'envoi")
            break

        # Convertir la ligne en dictionnaire
        message = row.to_dict()
        try:
            # Envoyer le message à Kafka
            producer.send(topic, value=message)
            print(f"Message {index + 1}/{len(data)} envoyé : {message}")
            last_index_processed = index + 1  # Mettre à jour l'index
        except Exception as e:
            print(f"Échec de l'envoi du message {index + 1}: {str(e)}")
            raise
        time.sleep(8)  # Délai pour tester

    # Sauvegarder l'état final
    save_last_index(last_index_processed)
    # S'assurer que tous les messages sont envoyés
    producer.flush()
    print("Tous les messages envoyés à Kafka")

except Exception as e:
    print(f"Échec du producteur : {str(e)}")
    raise

finally:
    # Fermer le producteur
    producer.close()
    print("Kafka producer fermé")