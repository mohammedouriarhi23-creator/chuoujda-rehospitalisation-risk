from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, from_json, udf, struct, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, FloatType, TimestampType
import joblib
import numpy as np
import os
import pandas as pd

spark = SparkSession.builder \
    .appName("DiabeticDataStreamingg") \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()


# Set log level to WARN to reduce noise
spark.sparkContext.setLogLevel("WARN")

model = joblib.load('./include/models/best_model.joblib')
encoder = joblib.load('./include/models/ordinal_encoder.pkl')

# Les diffuser (broadcast) pour les utiliser dans le UDF
broadcast_model = spark.sparkContext.broadcast(model)
broadcast_encoder = spark.sparkContext.broadcast(encoder)

input_columns = ['race', 'gender', 'age', 'discharge_disposition_id',
       'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
       'number_diagnoses', 'metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed']

cat_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 
             'metformin', 'repaglinide', 
            'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin',
            'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone', 'change', 'diabetesMed']

def predict_with_proba_udf(*cols):
    try:
        # Convert to numpy array
        features_raw = np.array(cols).reshape(1, -1)

        # Indices des colonnes catégorielles
        cat_indices = [input_columns.index(c) for c in cat_cols]

        # Séparer colonnes catégorielles et non-catégorielles
        cat_features = features_raw[:, cat_indices]
        num_indices = [i for i in range(len(input_columns)) if i not in cat_indices]
        num_features = features_raw[:, num_indices]

        # Encoder les colonnes catégorielles
        encoded_cat_features = broadcast_encoder.value.transform(cat_features)

        # Recombiner dans le bon ordre
        full_features = np.empty_like(features_raw, dtype=float)
        full_features[:, cat_indices] = encoded_cat_features
        full_features[:, num_indices] = num_features

        # Prédiction
        model = broadcast_model.value
        prediction = int(model.predict(full_features)[0])
        proba = model.predict_proba(full_features)[0]
        
        return (prediction, float(proba[0]), float(proba[1]))

    except Exception as e:
        print(f"UDF error: {str(e)}")
        return (-1, -1.0, -1.0)

prediction_schema = StructType([
    StructField("prediction", IntegerType(), False),
    StructField("proba_0", FloatType(), False),
    StructField("proba_1", FloatType(), False),
])

predict_udf_spark = udf(predict_with_proba_udf, prediction_schema)

kafka_schema = StructType([
    StructField("encounter_id", IntegerType(), True),
    StructField("patient_nbr", IntegerType(), True),
    StructField("race", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("age", StringType(), True),
    StructField("weight", StringType(), True),
    StructField("admission_type_id", IntegerType(), True),
    StructField("discharge_disposition_id", IntegerType(), True),
    StructField("admission_source_id", IntegerType(), True),
    StructField("time_in_hospital", IntegerType(), True),
    StructField("payer_code", StringType(), True),
    StructField("medical_specialty", StringType(), True),
    StructField("num_lab_procedures", IntegerType(), True),
    StructField("num_procedures", IntegerType(), True),
    StructField("num_medications", IntegerType(), True),
    StructField("number_outpatient", IntegerType(), True),
    StructField("number_emergency", IntegerType(), True),
    StructField("number_inpatient", IntegerType(), True),
    StructField("diag_1", StringType(), True),
    StructField("diag_2", StringType(), True),
    StructField("diag_3", StringType(), True),
    StructField("number_diagnoses", IntegerType(), True),
    StructField("max_glu_serum", StringType(), True),
    StructField("A1Cresult", StringType(), True),
    StructField("metformin", StringType(), True),
    StructField("repaglinide", StringType(), True),
    StructField("nateglinide", StringType(), True),
    StructField("chlorpropamide", StringType(), True),
    StructField("glimepiride", StringType(), True),
    StructField("acetohexamide", StringType(), True),
    StructField("glipizide", StringType(), True),
    StructField("glyburide", StringType(), True),
    StructField("tolbutamide", StringType(), True),
    StructField("pioglitazone", StringType(), True),
    StructField("rosiglitazone", StringType(), True),
    StructField("acarbose", StringType(), True),
    StructField("miglitol", StringType(), True),
    StructField("troglitazone", StringType(), True),
    StructField("tolazamide", StringType(), True),
    StructField("examide", StringType(), True),
    StructField("citoglipton", StringType(), True),
    StructField("insulin", StringType(), True),
    StructField("glyburide-metformin", StringType(), True),
    StructField("glipizide-metformin", StringType(), True),
    StructField("glimepiride-pioglitazone", StringType(), True),
    StructField("metformin-rosiglitazone", StringType(), True),
    StructField("metformin-pioglitazone", StringType(), True),
    StructField("change", StringType(), True),
    StructField("diabetesMed", StringType(), True),
])

def map_now():
    listname = [
        ('infections', 139),
        ('neoplasms', (239 - 139)),
        ('endocrine', (279 - 239)),
        ('blood', (289 - 279)),
        ('mental', (319 - 289)),
        ('nervous', (359 - 319)),
        ('sense', (389 - 359)),
        ('circulatory', (459 - 389)),
        ('respiratory', (519 - 459)),
        ('digestive', (579 - 519)),
        ('genitourinary', (629 - 579)),
        ('pregnancy', (679 - 629)),
        ('skin', (709 - 679)),
        ('musculoskeletal', (739 - 709)),
        ('congenital', (759 - 739)),
        ('perinatal', (779 - 759)),
        ('ill-defined', (799 - 779)),
        ('injury', (999 - 799))
    ]
    
    dictcout = {}
    count = 1
    for name, num in listname:
        for i in range(num):
            dictcout[str(count)] = name
            count += 1
    return dictcout

codes = map_now()

def codemap_udf(value):
    if value is None or value in ['unknown', '?']:
        return 'unknown'
    elif value.upper()[0] == 'V':
        return 'supplemental'
    elif value.upper()[0] == 'E':
        return 'injury'
    else:
        lkup = str(value).split('.')[0]
        return codes.get(lkup, 'unknown')

codemap_udf_spark = udf(codemap_udf, StringType())

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")) \
    .option("subscribe", os.environ.get("KAFKA_TOPIC", "hospital_fin67")) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), kafka_schema).alias("data")) \
    .select("data.*")
# 1. Drop columns: weight, payer_code, medical_specialty
df_transformed = df.drop("weight", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult")
# 3. Filter out specific discharge_disposition_id values
discharge_ids_to_exclude = [11, 13, 14, 19, 20, 21]
df_transformed = df_transformed.filter(
    ~col("discharge_disposition_id").isin(discharge_ids_to_exclude)
)


# 4. Fill numeric columns with 0
numeric_columns = ["time_in_hospital", "num_lab_procedures", "num_procedures", 
                   "num_medications", "number_outpatient", "number_emergency", 
                   "number_inpatient", "number_diagnoses", "admission_type_id", 
                   "discharge_disposition_id", "admission_source_id"]
for col_name in numeric_columns:
    df_transformed = df_transformed.withColumn(
        col_name, 
        when(col(col_name).isNull(), 0).otherwise(col(col_name))
    )

# 5. Fill object columns with 'unknown'
object_columns = ["race", "gender", "age", "diag_1", "diag_2", "diag_3", 
                   "metformin", "repaglinide", 
                  "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", 
                  "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
                  "rosiglitazone", "acarbose", "miglitol", "troglitazone", 
                  "tolazamide", "examide", "citoglipton", "insulin", 
                  "glyburide-metformin", "glipizide-metformin", 
                  "glimepiride-pioglitazone", "metformin-rosiglitazone", 
                  "metformin-pioglitazone", "change", "diabetesMed"]
for col_name in object_columns:
    df_transformed = df_transformed.withColumn(
        col_name, 
        when(col(col_name).isNull() | (col(col_name) == "?") | (col(col_name) == '"NaN"'), "unknown").otherwise(col(col_name))
    )

# 6. Encode diagnosis columns using the codemap_udf
diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
for col_name in diagnosis_columns:
    df_transformed = df_transformed.withColumn(col_name, codemap_udf_spark(col(col_name)))

# 7. Drop additional columns: encounter_id, patient_nbr, admission_type_id, readmitted
df_final = df_transformed.drop("encounter_id","admission_type_id")

# Add predictions and timestamp
df_result = df_final.withColumn(
    "prediction_struct",
    predict_udf_spark(*[col(c) for c in input_columns])
).withColumn("timestamp", current_timestamp())

# Séparer les colonnes
df_result = df_result.select(
    "*",
    col("prediction_struct.prediction").alias("prediction"),
    col("prediction_struct.proba_0"),
    col("prediction_struct.proba_1")
).drop("prediction_struct")

db_name = os.environ.get("MYSQL_DATABASE", "ma_base_mysql")
mysql_url = f"jdbc:mysql://mysql:3306/{db_name}?useSSL=false&allowPublicKeyRetrieval=true"
nom_table_test = "table_tessst"
utilisateur_mysql = os.environ.get("MYSQL_USER", "mon_utilisateur_mysql")
mot_de_passe_mysql = os.environ.get("MYSQL_PASSWORD", "Dracule@8156")
def ecrire_dans_mysql(batch_df, batch_id):
    batch_df.write \
        .format("jdbc") \
        .option("url", mysql_url) \
        .option("dbtable", nom_table_test) \
        .option("user", utilisateur_mysql) \
        .option("password", mot_de_passe_mysql) \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .mode("append") \
        .save()

# Utiliser writeStream avec foreachBatch et un trigger
mysql_query = df_result.writeStream \
    .foreachBatch(ecrire_dans_mysql) \
    .outputMode("append") \
    .start()

import time
# Enforce termination after 2 minutes
start_time = time.time()
timeout_seconds = 50

try:
    while mysql_query.isActive and (time.time() - start_time) < timeout_seconds:
        time.sleep(1)  # Check every second
    if mysql_query.isActive:
        print("Timeout reached, stopping the streaming query...")
        mysql_query.stop()  # Explicitly stop the query after 2 minutes
except KeyboardInterrupt:
    print("Received KeyboardInterrupt, stopping the streaming query...")
    mysql_query.stop()
finally:
    spark.stop()
    print("Spark session stopped")