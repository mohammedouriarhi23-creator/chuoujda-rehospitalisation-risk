import os
import signal
import subprocess
import shutil
import time

def kill_processes_by_name(name):
    try:
        # Trouver les processus contenant le nom
        result = subprocess.check_output(['ps', 'aux']).decode()
        for line in result.splitlines():
            if name in line and 'grep' not in line:
                pid = int(line.split()[1])
                print(f"Killing process {pid} ({name})")
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error killing processes: {e}")

def remove_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")

if __name__ == "__main__":
    # Tuer les processus Spark
    kill_processes_by_name('org.apache.spark.deploy')
    kill_processes_by_name('spark_consumer.py')

    # Supprimer les checkpoints Spark
    remove_path('/tmp/spark-checkpoint')

    # Supprimer le flag de cleanup
    remove_path('./include/tmp/cleanup_flag.txt')

    print("Cleanup completed.")
    time.sleep(10)
