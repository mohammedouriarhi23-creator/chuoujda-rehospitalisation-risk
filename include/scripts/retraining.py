import mysql.connector
import pandas as pd
import joblib
import os
import numpy as np
import mlflow
import mlflow.sklearn
from prometheus_client import Gauge, start_http_server
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import json
warnings.filterwarnings('ignore')

# Configurer MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Model_Training_Experiment"))

def feature_engineering(data):
    data = data[data['readmitted'] != '>30']
    missing_ratio = data.apply(lambda x: ((x == '?').sum() + x.isnull().sum()) / len(x))
    missing_threshold = 0.3
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    data = data.drop(columns=columns_to_drop)
    print("\nColonnes supprimées pour valeurs manquantes ou '?' élevées:", columns_to_drop)
    data['readmitted'] = np.where(data['readmitted'] == 'NO', 0, 1)
    data = data[((data.discharge_disposition_id != 11) & 
                 (data.discharge_disposition_id != 13) &
                 (data.discharge_disposition_id != 14) & 
                 (data.discharge_disposition_id != 19) & 
                 (data.discharge_disposition_id != 20) & 
                 (data.discharge_disposition_id != 21))]
    data = data.replace('?', np.nan)
    numcolumn = data.select_dtypes(include=[np.number]).columns
    objcolumn = data.select_dtypes(include=['object']).columns
    data[numcolumn] = data[numcolumn].fillna(0)
    data[objcolumn] = data[objcolumn].fillna("unknown")

    def map_now():
        listname = [
            ('infections', 139), ('neoplasms', (239 - 139)), ('endocrine', (279 - 239)),
            ('blood', (289 - 279)), ('mental', (319 - 289)), ('nervous', (359 - 319)),
            ('sense', (389 - 359)), ('circulatory', (459-389)), ('respiratory', (519-459)),
            ('digestive', (579 - 519)), ('genitourinary', (629 - 579)), ('pregnancy', (679 - 629)),
            ('skin', (709 - 679)), ('musculoskeletal', (739 - 709)), ('congenital', (759 - 739)),
            ('perinatal', (779 - 759)), ('ill-defined', (799 - 779)), ('injury', (999 - 799))
        ]
        dictcout = {}
        count = 1
        for name, num in listname:
            for i in range(num):
                dictcout.update({str(count): name})
                count += 1
        return dictcout

    def codemap(df, codes):
        namecol = df.columns.tolist()
        for col in namecol:
            temp = []
            for num in df[col]:
                if ((num is None) | (num in ['unknown', '?']) | (pd.isnull(num))):
                    temp.append('unknown')
                elif num.upper()[0] == 'V':
                    temp.append('supplemental')
                elif num.upper()[0] == 'E':
                    temp.append('injury')
                else:
                    lkup = num.split('.')[0]
                    temp.append(codes[lkup])
            df.loc[:, col] = temp
        return df

    listcol = ['diag_1', 'diag_2', 'diag_3']
    codes = map_now()
    data[listcol] = codemap(data[listcol], codes)
    data = data.drop(['encounter_id', 'admission_type_id'], axis=1)
    return data

# Charger l'historique cumulé
cumulative_path = './include/data/historical_data_cumulative.csv'
if os.path.exists(cumulative_path):
    df_cumulative = pd.read_csv(cumulative_path)
else:
    df_cumulative = pd.read_csv('./include/data/historical_data.csv')
    df_cumulative = feature_engineering(df_cumulative)

# Extraire les nouvelles données de MySQL
config = {
    'user': os.environ.get('MYSQL_USER', 'mon_utilisateur_mysql'),
    'password': os.environ.get('MYSQL_PASSWORD', 'Dracule@8156'),
    'host': 'mysql',
    'port': 3306,
    'database': os.environ.get('MYSQL_DATABASE', 'ma_base_mysql'),
}
conn = mysql.connector.connect(**config)
query = "SELECT * FROM table_tessst"
df_new = pd.read_sql(query, conn)
df_new = df_new.drop(['timestamp', "proba_0", 'proba_1'], axis=1)
df_new = df_new.rename(columns={"prediction": "readmitted"})
conn.close()

# Filtrer les doublons
if 'patient_nbr' in df_new.columns and 'patient_nbr' in df_cumulative.columns:
    df_new = df_new[~df_new['patient_nbr'].isin(df_cumulative['patient_nbr'])]

# Ajouter les nouvelles données à l'historique cumulé
df_cumulative = pd.concat([df_cumulative, df_new], ignore_index=True)
print("Lignes avec NaN dans readmitted:\n", df_cumulative[df_cumulative['readmitted'].isna()])
df_cumulative.to_csv(cumulative_path, index=False)

# Normalisation des données
listnormal = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
              'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
normal = StandardScaler()
df_cumulative[listnormal] = normal.fit_transform(df_cumulative[listnormal])

# Encodage des colonnes catégorielles
cat_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
            'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']

def ordinal_encoding(df, cat_cols):
    df_encoded = df.copy()
    encoder = OrdinalEncoder()
    df_encoded[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df_encoded, encoder

dff, encoder = ordinal_encoding(df_cumulative, cat_cols)

# Préparer les features et la cible
X = dff.drop(columns=['readmitted', 'patient_nbr'])
y = dff['readmitted'].astype(int)
print("Nombre de NaN dans y avant nettoyage:", dff.readmitted.value_counts())

# Suréchantillonnage avec SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)
from sklearn.metrics import precision_score
import time

# Fonctions d'entraînement des modèles
def train_random_forest(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Random_Forest"):
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        # Calcul du taux de faux positifs (approximation via matrice de confusion si besoin)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.sklearn.log_model(rf, "model")
        
        return rf, accuracy, auc, recall, precision, false_positive_rate, training_time
def train_lightgbm(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="LightGBM"):
        start_time = time.time()
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.1, 'feature_fraction': 0.9,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgb_params, train_data, num_boost_round=100, valid_sets=[train_data], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        training_time = time.time() - start_time
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Enregistrer le modèle
        mlflow.lightgbm.log_model(model, "model")

        print(f"LightGBM - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, False Positive Rate: {false_positive_rate:.4f}, Training Time: {training_time:.4f}")
        return model, accuracy, auc, recall, precision, false_positive_rate, training_time

def train_xgboost(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="XGBoost"):
        start_time = time.time()
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6,
            'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Enregistrer le modèle
        mlflow.xgboost.log_model(model, "model")
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, False Positive Rate: {false_positive_rate:.4f}, Training Time: {training_time:.4f}")
        return model, accuracy, auc, recall, precision, false_positive_rate, training_time

def train_logistic_regression(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Logistic_Regression"):
        start_time = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr = LogisticRegression(random_state=42, max_iter=500, solver='liblinear', C=1.0)
        lr.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        y_pred = lr.predict(X_test_scaled)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Enregistrer le modèle
        mlflow.sklearn.log_model(lr, "model")
        
       

        print(f"Logistic Regression - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, False Positive Rate: {false_positive_rate:.4f}, Training Time: {training_time:.4f}")
        return lr, accuracy, auc, recall, precision, false_positive_rate, training_time

def train_decision_tree(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Decision_Tree"):
        start_time = time.time()
        dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42)
        dt.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = dt.predict(X_test)
        y_pred_proba = dt.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Enregistrer les métriques dans MLflow
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("training_time_seconds", training_time)

        
        # Enregistrer le modèle
        mlflow.sklearn.log_model(dt, "model")
        
       

        print(f"Decision Tree - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, False Positive Rate: {false_positive_rate:.4f}, Training Time: {training_time:.4f}")
        return dt, accuracy, auc, recall, precision, false_positive_rate, training_time

def train_all_models(X_train, y_train, X_test, y_test):
    results = {}
    try:
        rf_model, rf_acc, rf_auc, rf_recall, rf_precision, rf_false_positive_rate, rf_training_time = train_random_forest(X_train, y_train, X_test, y_test)
        results['Random Forest'] = {'model': rf_model, 'accuracy': rf_acc, 'auc': rf_auc, 'recall': rf_recall, 'precision': rf_precision, 'false_positive_rate': rf_false_positive_rate, 'training_time': rf_training_time}
    except Exception as e:
        print(f"Erreur Random Forest: {e}")
    try:
        lgb_model, lgb_acc, lgb_auc, lgb_recall, lgb_precision, lgb_false_positive_rate, lgb_training_time = train_lightgbm(X_train, y_train, X_test, y_test)
        results['LightGBM'] = {'model': lgb_model, 'accuracy': lgb_acc, 'auc': lgb_auc, 'recall': lgb_recall, 'precision': lgb_precision, 'false_positive_rate': lgb_false_positive_rate, 'training_time': lgb_training_time}
    except Exception as e:
        print(f"Erreur LightGBM: {e}")
    try:
        xgb_model, xgb_acc, xgb_auc, xgb_recall, xgb_precision, xgb_false_positive_rate, xgb_training_time = train_xgboost(X_train, y_train, X_test, y_test)
        results['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_acc, 'auc': xgb_auc, 'recall': xgb_recall, 'precision': xgb_precision, 'false_positive_rate': xgb_false_positive_rate, 'training_time': xgb_training_time}
    except Exception as e:
        print(f"Erreur XGBoost: {e}")
    try:
        lr_model, lr_acc, lr_auc, lr_recall, lr_precision, lr_false_positive_rate, lr_training_time = train_logistic_regression(X_train, y_train, X_test, y_test)
        results['Logistic Regression'] = {'model': lr_model, 'accuracy': lr_acc, 'auc': lr_auc, 'recall': lr_recall, 'precision': lr_precision, 'false_positive_rate': lr_false_positive_rate, 'training_time': lr_training_time}
    except Exception as e:
        print(f"Erreur Logistic Regression: {e}")
    try:
        dt_model, dt_acc, dt_auc, dt_recall, dt_precision, dt_false_positive_rate, dt_training_time = train_decision_tree(X_train, y_train, X_test, y_test)
        results['Decision Tree'] = {'model': dt_model, 'accuracy': dt_acc, 'auc': dt_auc, 'recall': dt_recall, 'precision': dt_precision, 'false_positive_rate': dt_false_positive_rate, 'training_time': dt_training_time}
    except Exception as e:
        print(f"Erreur Decision Tree: {e}")
    return results

def compare_and_save_best_model(results, recall_threshold=0.02, auc_threshold=0.01, acc_threshold=0.01):
    with mlflow.start_run(run_name="Best_Model"):
        scores = []
        for name, res in results.items():
            scores.append({
                'name': name,
                'model': res['model'],
                'accuracy': res['accuracy'],
                'auc': res['auc'],
                'recall': res['recall'],
                'precision': res['precision'],
                'false_positive_rate': res['false_positive_rate'],
                'training_time': res['training_time']
            })
        
        scores.sort(key=lambda x: x['recall'], reverse=True)
        best = scores[0]
        for current in scores[1:]:
            if abs(current['recall'] - best['recall']) < recall_threshold:
                if current['auc'] > best['auc'] + auc_threshold:
                    best = current
                elif abs(current['auc'] - best['auc']) < auc_threshold:
                    if current['accuracy'] > best['accuracy'] + acc_threshold:
                        best = current
        
        print(f"\n>>> Meilleur modèle: {best['name']}")
        print(f"Recall: {best['recall']:.4f}, AUC: {best['auc']:.4f}, Accuracy: {best['accuracy']:.4f}")
        
        # Enregistrer le meilleur modèle dans MLflow
        mlflow.log_param("best_model", best['name'])
        mlflow.log_metric("best_recall", best['recall'])
        mlflow.log_metric("best_auc", best['auc'])
        mlflow.log_metric("best_accuracy", best['accuracy'])
        mlflow.log_metric("best_precision", best['precision'])
        mlflow.log_metric("best_false_positive_rate", best['false_positive_rate'])
        mlflow.log_metric("best_training_time", best['training_time'])

        all_metrics = {model_name: {'accuracy': data['accuracy'], 'auc': data['auc'], 'recall': data['recall'],
                           'precision': data.get('precision', 0), 'false_positive_rate': data.get('false_positive_rate', 0),
                           'training_time_seconds': data.get('training_time', 0)}  # Remplace 'training_time_seconds' par 'training_time'
              for model_name, data in results.items()}
        with open('/usr/local/airflow/include/scripts/metrics.json', 'w') as f:
            json.dump(all_metrics, f)

results = train_all_models(X_train, y_train, X_test, y_test)
compare_and_save_best_model(results)