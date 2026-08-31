from airflow.decorators import dag, task
from datetime import datetime, timedelta
import subprocess
import time
import os
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DAG_ID = 'readmission_pipeline'

@dag(
    DAG_ID,
    schedule=None,
    catchup=False,
)
def readmission_pipeline():
    from airflow.operators.bash import BashOperator

    cleanup_task = BashOperator(
        task_id='cleanup_resources',
        bash_command='python3 /usr/local/airflow/include/scripts/cleanup_resources.py',
    )

    def create_kafka_topic():
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        topic_name = os.environ.get('KAFKA_TOPIC', 'hospital_fin67')
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
            existing = admin.list_topics()
            if topic_name not in existing:
                admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
                print(f"Topic {topic_name} created")
            else:
                print(f"Topic {topic_name} already exists")
            admin.close()
        except Exception as e:
            print(f"Topic creation: {e}")

    create_topic_task = PythonOperator(
        task_id='create_kafka_topic',
        python_callable=create_kafka_topic,
    )

    def run_producer_and_spark():
        """Run producer in background, then spark consumer, both in parallel."""
        import threading

        def run_producer():
            try:
                subprocess.run(['python', 'include/scripts/kafka_producer.py'], check=True)
            except Exception as e:
                print(f"Error in producer: {e}")

        # Start producer in a background thread
        producer_thread = threading.Thread(target=run_producer, daemon=True)
        producer_thread.start()

        # Give producer time to send first messages
        time.sleep(15)

        # Run spark consumer (blocking)
        try:
            subprocess.run([
                'spark-submit',
                '--master', 'spark://spark-master:7077',
                '--jars', '/usr/local/airflow/include/jars/mysql-connector-j-8.4.0.jar,/usr/local/airflow/include/jars/spark-sql-kafka-0-10_2.12-3.4.4.jar,/usr/local/airflow/include/jars/kafka-clients-2.8.1.jar,/usr/local/airflow/include/jars/spark-streaming_2.12-3.4.4.jar,/usr/local/airflow/include/jars/spark-token-provider-kafka-0-10_2.12-3.4.4.jar,/usr/local/airflow/include/jars/commons-pool2-2.11.1.jar',
                '--conf', 'spark.driver.extraClassPath=/usr/local/airflow/include/jars/mysql-connector-j-8.4.0.jar:/usr/local/airflow/include/jars/spark-sql-kafka-0-10_2.12-3.4.4.jar:/usr/local/airflow/include/jars/kafka-clients-2.8.1.jar:/usr/local/airflow/include/jars/spark-streaming_2.12-3.4.4.jar:/usr/local/airflow/include/jars/spark-token-provider-kafka-0-10_2.12-3.4.4.jar:/usr/local/airflow/include/jars/commons-pool2-2.11.1.jar',
                '--conf', 'spark.pyspark.python=python3',
                '--conf', 'spark.pyspark.driver.python=python3',
                'include/scripts/spark_consumer.py'
            ], check=True)
        except Exception as e:
            print(f"Error in spark consumer: {e}")
            raise

        # Wait for producer to finish
        producer_thread.join(timeout=30)

    streaming_task = PythonOperator(
        task_id='run_streaming_pipeline',
        python_callable=run_producer_and_spark,
    )

    def check_services():
        import requests
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        try:
            mlflow_response = requests.get(mlflow_uri, timeout=10)
            prometheus_response = requests.get("http://prometheus:9090", timeout=10)
            if mlflow_response.status_code < 500 and prometheus_response.status_code < 500:
                print(f"MLflow: {mlflow_response.status_code}, Prometheus: {prometheus_response.status_code} - Services are running")
            else:
                raise Exception(f"Services not available - MLflow: {mlflow_response.status_code}, Prometheus: {prometheus_response.status_code}")
        except requests.ConnectionError as e:
            raise Exception(f"Service check failed: {e}")

    check_services_task = PythonOperator(
        task_id='check_services',
        python_callable=check_services,
    )

    def run_retraining():
        import subprocess
        try:
            subprocess.run(['python', '/usr/local/airflow/include/scripts/retraining.py'], check=True)
        except Exception as e:
            print(f"Error in retraining: {e}")
            raise

    retraining_task = PythonOperator(
        task_id='run_retraining',
        python_callable=run_retraining,
    )

    trigger_next = TriggerDagRunOperator(
        task_id='trigger_next_run',
        trigger_dag_id=DAG_ID,
        wait_for_completion=False,
    )

    # Set up the workflow
    cleanup_task >> create_topic_task >> streaming_task >> check_services_task >> retraining_task >> trigger_next

readmission_pipeline()
