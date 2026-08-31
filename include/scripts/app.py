from flask import Flask, render_template_string, request, redirect, url_for
import os
import google.generativeai as genai
import pandas as pd

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# Load streaming.csv dataset for context
try:
    df = pd.read_csv('/usr/local/airflow/include/data/streaming_data.csv')
    dataset_context = df.describe().to_string() + "\n\nSample data:\n" + df.head().to_string()
except FileNotFoundError:
    dataset_context = "Streaming dataset not found. Using general medical knowledge."

# Initialize Gemini model
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Template HTML pour la page d'accueil
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Professionnel - Médecine, IT & Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            flex: 1;
        }

        header {
            text-align: center;
            padding: 60px 0 40px;
            color: white;
        }

        .welcome-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 50px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .welcome-title {
            font-size: 3.5rem;
            font-weight: 300;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .welcome-subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 30px;
            line-height: 1.6;
        }

        .buttons-container {
            display: flex;
            justify-content: center;
            gap: 60px;
            flex-wrap: wrap;
            margin-top: 60px;
        }

        .dashboard-button {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 300px;
            height: 200px;
            background: rgba(255, 255, 255, 0.95);
            border: none;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            color: #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }

        .dashboard-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.5s;
        }

        .dashboard-button:hover::before {
            left: 100%;
        }

        .dashboard-button:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }

        .medecin-button {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
        }

        .it-button {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            color: white;
        }

        .chatbot-button {
            background: linear-gradient(135deg, #6ab04c, #55a630);
            color: white;
        }

        .button-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
        }

        .button-text {
            font-size: 1.5rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .button-description {
            font-size: 0.9rem;
            margin-top: 10px;
            opacity: 0.8;
            text-align: center;
            padding: 0 20px;
        }

        footer {
            text-align: center;
            padding: 30px 0;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .welcome-title {
                font-size: 2.5rem;
            }
            
            .buttons-container {
                gap: 30px;
            }
            
            .dashboard-button {
                width: 250px;
                height: 180px;
            }
            
            .button-icon {
                font-size: 3rem;
            }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="welcome-section">
                <h1 class="welcome-title">Bienvenue</h1>
                <p class="welcome-subtitle">
                    Plateforme professionnelle de visualisation de données et d'assistance médicale<br>
                    Accédez à vos dashboards spécialisés ou à l'assistant médical en un clic
                </p>
            </div>
        </header>

        <main>
            <div class="buttons-container">
                <a href="{{ url_for('medecin_dashboard') }}" class="dashboard-button medecin-button pulse">
                    <div class="button-icon">
                        <i class="fas fa-heartbeat"></i>
                    </div>
                    <div class="button-text">Médecine</div>
                    <div class="button-description">
                        Prédictions, alertes et analyses médicales
                    </div>
                </a>

                <a href="{{ url_for('it_dashboard') }}" class="dashboard-button it-button pulse">
                    <div class="button-icon">
                        <i class="fas fa-server"></i>
                    </div>
                    <div class="button-text">IT</div>
                    <div class="button-description">
                        MLflow et monitoring des systèmes
                    </div>
                </a>

                <a href="{{ url_for('chatbot') }}" class="dashboard-button chatbot-button pulse">
                    <div class="button-icon">
                        <i class="fas fa-comment-medical"></i>
                    </div>
                    <div class="button-text">Assistant Médical</div>
                    <div class="button-description">
                        Chatbot pour analyses et conseils médicaux
                    </div>
                </a>
            </div>
        </main>

        <footer>
            <p>© 2025 Dashboard Professionnel. Tous droits réservés.</p>
        </footer>
    </div>
</body>
</html>
"""

# Template pour le dashboard médecine
MEDECIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Médecine - Analyses & Prédictions</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            min-height: 100vh;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
        }

        .header-title {
            color: white;
            font-size: 2rem;
            font-weight: 300;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .back-button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .back-button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-icon {
            font-size: 3rem;
            color: #ff6b6b;
            margin-bottom: 15px;
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }

        .stat-label {
            color: #666;
            font-size: 1.1rem;
        }

        .grafana-container {
            background: transparent;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .grafana-title {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .grafana-frame {
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 10px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: #666;
        }

        .alert-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .alert-item {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .alert-critical {
            background: #f8d7da;
            border-left-color: #dc3545;
        }

        .alert-warning {
            background: #d4edda;
            border-left-color: #28a745;
        }

        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 20px;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1 class="header-title">
                <i class="fas fa-heartbeat"></i>
                Dashboard Médecine
            </h1>
            <a href="{{ url_for('home') }}" class="back-button">
                <i class="fas fa-arrow-left"></i>
                Retour
            </a>
        </div>
    </div>

    <div class="dashboard-container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-users"></i>
                </div>
                <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=now-1m&to=now&&timezone=browser&refresh=1s&theme=light&panelId=8&__feature.dashboardSceneSolo" width="350" height="200" frameborder="0"></iframe>
            </div>
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751275490388&to=1751279090388&timezone=browser&refresh=auto&theme=light&panelId=9&__feature.dashboardSceneSolo" width="350" height="200" frameborder="0"></iframe>
            </div>
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-brain"></i>
                </div>
                <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751275632119&to=1751279232119&timezone=browser&refresh=auto&theme=light&panelId=10&__feature.dashboardSceneSolo" width="350" height="200" frameborder="0"></iframe>
            </div>
        </div>

        <div class="grafana-container">
            <h2 class="grafana-title">
                <i class="fas fa-chart-area"></i>
                Analyses des réadmissions
            </h2>
            <div class="grafana-grid">
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-percentage"></i>
                        Comparer les réadmissions selon age
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751276716162&to=1751280316162&timezone=browser&refresh=auto&panelId=13&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-exclamation-triangle"></i>
                        Comparer les réadmissions selon race
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751276534640&to=1751280134640&timezone=browser&refresh=auto&panelId=12&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            <div class="grafana-grid">
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-percentage"></i>
                        Les patients réadmis au cours des dernières 30 heures
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751278990890&to=1751280790890&timezone=browser&refresh=auto&panelId=14&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-exclamation-triangle"></i>
                        Analyser l'évolution des réadmissions sur une période (par exemple, les 30 derniers jours)
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751195920088&to=1751282320088&timezone=browser&refresh=auto&theme=light&panelId=15&__feature.dashboardSceneSolo" width="1000" height="270" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
        </div>

        <div class="grafana-container">
            <h2 class="grafana-title">
                <i class="fas fa-hospital-user"></i>
                Analyse des Patients
            </h2>
            <div class="grafana-grid">
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-percentage"></i>
                        Répartition des patients par race
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751246549561&to=1751250149561&timezone=browser&refresh=auto&panelId=1&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-exclamation-triangle"></i>
                        Répartition des patients par genre
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751246600532&to=1751250200532&timezone=browser&refresh=auto&panelId=2&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            <div class="grafana-grid">
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-pills"></i>
                        Répartition des patients par age
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751246718648&to=1751250318648&timezone=browser&refresh=auto&panelId=3&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-door-open"></i>
                        Temps moyen d'hospitalisation par race
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751246809723&to=1751250409723&timezone=browser&refresh=auto&panelId=4&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            <div class="grafana-grid">
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-vial"></i>
                        Fréquence d'utilisation de l'insuline
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751246994821&to=1751250594821&timezone=browser&refresh=auto&panelId=5&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item">
                    <h3 class="grafana-subtitle">
                        <i class="fas fa-calculator"></i>
                        Fréquence d'utilisation des autres médicaments
                    </h3>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751248979844&to=1751252579844&timezone=browser&refresh=auto&panelId=6&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            <div class="grafana-container">
                <h3 class="grafana-subtitle">
                    <i class="fas fa-calculator"></i>
                    Fréquence des diagnostics principaux
                </h3>
                <div class="iframe-bg-black">
                    <iframe src="http://localhost:3000/d-solo/ac039751-74d1-4fe6-bd22-685495aaa6d8/pfa?orgId=1&from=1751247575090&to=1751251175090&timezone=browser&refresh=auto&panelId=7&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                </div>
            </div>
        </div>

        <style>
            .grafana-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }

            .grafana-item {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                overflow: hidden;
            }

            .grafana-subtitle {
                font-size: 1.2rem;
                color: #333;
                padding: 15px;
                margin: 0;
                background: rgba(255, 255, 255, 0.1);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .grafana-subtitle i {
                color: #ff6b6b;
            }

            @media (max-width: 968px) {
                .grafana-grid {
                    grid-template-columns: 1fr;
                }
            }

            .iframe-bg-black {
                background: #111;
                border-radius: 16px;
                padding: 16px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.15);
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>

        <div class="alert-section">
            <h2 class="grafana-title">
                <i class="fas fa-bell"></i>
                Alertes Médicales
            </h2>
            <div class="alert-item alert-critical">
                <i class="fas fa-exclamation-circle" style="color: #dc3545;"></i>
                <div>
                    <strong>Alerte Critique:</strong> Pic de glycémie détecté chez 3 patients
                </div>
            </div>
            <div class="alert-item alert-critical">
                <i class="fas fa-exclamation-circle" style="color: #dc3545;"></i>
                <div>
                    <strong>Réadmissions Urgentes:</strong> 5 patients réadmis en urgence dans les dernières 24h
                </div>
            </div>
            <div class="alert-item alert-critical">
                <i class="fas fa-exclamation-circle" style="color: #dc3545;"></i>
                <div>
                    <strong>Diagnostic Critique:</strong> 3 réadmissions avec complications cardiaques sévères
                </div>
            </div>
            <div class="alert-item alert-warning">
                <i class="fas fa-exclamation-triangle" style="color: #ffc107;"></i>
                <div>
                    <strong>Multi-Réadmissions:</strong> 2 patients réadmis plus de 3 fois en 30 jours
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Template pour le dashboard IT
IT_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard IT - MLflow & Monitoring</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            min-height: 100vh;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
        }

        .header-title {
            color: white;
            font-size: 2rem;
            font-weight: 300;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .back-button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .back-button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
        }

        .grafana-container {
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .grafana-title {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .grafana-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }

        .grafana-item {
            border-radius: 16px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .grafana-item.bg1 { background: #e0f7fa; }
        .grafana-item.bg2 { background: #ffe5ec; }
        .grafana-item.bg3 { background: #f3ffe3; }
        .grafana-item.bg4 { background: #f7e8ff; }
        .grafana-item.bg5 { background: #fffbe5; }

        .grafana-subtitle {
            font-size: 1.2rem;
            color: #333;
            padding: 18px 20px;
            background: rgba(255,255,255,0.25);
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }

        .grafana-subtitle i {
            color: #4ecdc4;
        }

        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 20px;
            }
            .grafana-grid {
                grid-template-columns: 1fr;
            }
        }

        .iframe-bg-black {
            background: #111;
            border-radius: 0 0 16px 16px;
            padding: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1 class="header-title">
                <i class="fas fa-server"></i>
                Dashboard IT
            </h1>
            <a href="{{ url_for('home') }}" class="back-button">
                <i class="fas fa-arrow-left"></i>
                Retour
            </a>
        </div>
    </div>

    <div class="dashboard-container">
        <div class="grafana-container">
            <h2 class="grafana-title">
                <i class="fas fa-chart-line"></i>
                Métriques de Performance des Modèles
            </h2>
            <div class="grafana-grid">
                <div class="grafana-item bg5">
                    <div class="grafana-subtitle">
                        <i class="fas fa-check-circle"></i>
                        Accuracy
                    </div>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/49ac696b-622c-4ff1-a50d-f47087da5468/pfa-models?orgId=1&from=now-5m&to=now&refresh=5s&timezone=browser&panelId=1&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item bg2">
                    <div class="grafana-subtitle">
                        <i class="fas fa-exclamation-triangle"></i>
                        False Positives
                    </div>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/49ac696b-622c-4ff1-a50d-f47087da5468/pfa-models?orgId=1&from=now-5m&to=now&timezone=browser&refresh=5s&panelId=4&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item bg3">
                    <div class="grafana-subtitle">
                        <i class="fas fa-bullseye"></i>
                        Précision
                    </div>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/49ac696b-622c-4ff1-a50d-f47087da5468/pfa-models?orgId=1&from=now-5m&to=now&timezone=browser&refresh=5s&panelId=3&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                <div class="grafana-item bg4">
                    <div class="grafana-subtitle">
                        <i class="fas fa-search"></i>
                        Recall
                    </div>
                    <div class="iframe-bg-black">
                        <iframe src="http://localhost:3000/d-solo/49ac696b-622c-4ff1-a50d-f47087da5468/pfa-models?orgId=1&from=now-5m&to=now&timezone=browser&refresh=5s&panelId=2&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
                 <div class="grafana-item bg4">
                    <div class="grafana-subtitle">
                        <i class="fas fa-search"></i>
                        AUC Score
                    </div>
                    <div class="iframe-bg-black">
<iframe src="http://localhost:3000/d-solo/49ac696b-622c-4ff1-a50d-f47087da5468/pfa-models?orgId=1&from=now-5m&to=now&timezone=browser&refresh=5s&panelId=5&__feature.dashboardSceneSolo" width="1000" height="300" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #666; font-size: 0.9rem;">
                <i class="fas fa-info-circle"></i>
                Métriques de performance : AUC-ROC (performance globale), False Positives (erreurs), Précision (exactitude), Recall (sensibilité) et Accuracy (précision globale)
            </div>
        </div>
    </div>
</body>
</html>
"""

# Template pour le chatbot
CHATBOT_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistant Médical - Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #6ab04c, #55a630);
            min-height: 100vh;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
        }

        .header-title {
            color: white;
            font-size: 2rem;
            font-weight: 300;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .back-button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .back-button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }

        .chatbot-container {
            max-width: 800px;
            margin: 40px auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            height: 600px;
        }

        .chatbot-header {
            background: linear-gradient(135deg, #6ab04c, #55a630);
            color: white;
            padding: 15px;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            border-radius: 10px 10px 0 0;
        }

        .chatbot-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .chatbot-message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
            line-height: 1.4;
        }

        .chatbot-message.user {
            background: #6ab04c;
            color: white;
            margin-left: auto;
        }

        .chatbot-message.bot {
            background: #e9ecef;
            color: #333;
        }

        .chatbot-input-container {
            display: flex;
            padding: 15px;
            background: #fff;
            border-top: 1px solid #ddd;
        }

        .chatbot-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 20px;
            outline: none;
            font-size: 1rem;
        }

        .chatbot-send {
            background: #6ab04c;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            margin-left: 10px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .chatbot-send:hover {
            background: #55a630;
        }

        @media (max-width: 768px) {
            .chatbot-container {
                margin: 20px;
                height: 500px;
            }

            .header-content {
                flex-direction: column;
                gap: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1 class="header-title">
                <i class="fas fa-comment-medical"></i>
                Assistant Médical
            </h1>
            <a href="{{ url_for('home') }}" class="back-button">
                <i class="fas fa-arrow-left"></i>
                Retour
            </a>
        </div>
    </div>

    <div class="chatbot-container">
        <div class="chatbot-header">
            Assistant Médical - Posez vos questions
        </div>
        <div class="chatbot-messages" id="chatbot-messages">
            <div class="chatbot-message bot">
                Bonjour ! Je suis votre assistant médical. Posez-moi des questions sur les données des patients ou les analyses de réadmission.
            </div>
        </div>
        <div class="chatbot-input-container">
            <input type="text" class="chatbot-input" id="chatbot-input" placeholder="Posez votre question...">
            <button class="chatbot-send" onclick="window.sendMessage()">Envoyer</button>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        window.sendMessage = async function() {
            const input = document.getElementById('chatbot-input');
            const messages = document.getElementById('chatbot-messages');
            const message = input.value.trim();
            if (!message) return;

            // Add user message
            const userMessage = document.createElement('div');
            userMessage.className = 'chatbot-message user';
            userMessage.textContent = message;
            messages.appendChild(userMessage);
            input.value = '';

            // Scroll to bottom
            messages.scrollTop = messages.scrollHeight;

            // Send message to server
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                
                // Add bot response
                const botMessage = document.createElement('div');
                botMessage.className = 'chatbot-message bot';
                botMessage.textContent = data.response;
                messages.appendChild(botMessage);
                
                // Scroll to bottom
                messages.scrollTop = messages.scrollHeight;
            } catch (error) {
                console.error('Error:', error);
                const errorMessage = document.createElement('div');
                errorMessage.className = 'chatbot-message bot';
                errorMessage.textContent = `Erreur lors de la communication avec l'assistant.`;
                messages.appendChild(errorMessage);
                messages.scrollTop = messages.scrollHeight;
            }
        }

        // Allow sending message with Enter key
        document.getElementById('chatbot-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                window.sendMessage();
            }
        });
    });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Page d'accueil avec les boutons Médecine, IT et Chatbot"""
    return render_template_string(HOME_TEMPLATE)

@app.route('/medecin')
def medecin_dashboard():
    """Dashboard Grafana pour la section Médecine"""
    return render_template_string(MEDECIN_TEMPLATE)

@app.route('/it')
def it_dashboard():
    """Dashboard Grafana pour la section IT avec MLflow"""
    return render_template_string(IT_TEMPLATE)

@app.route('/chatbot')
def chatbot():
    """Page pour l'assistant médical chatbot"""
    return render_template_string(CHATBOT_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot requests"""
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Prepare prompt with dataset context
    prompt = f"""
    Vous êtes un assistant médical utilisant les données du fichier streaming.csv. Voici un résumé des données :
    {dataset_context}
    
    Répondez à la question suivante en tant qu'assistant médical, en vous basant sur les données ci-dessus et vos connaissances médicales générales si nécessaire :
    {user_message}
    """
    
    try:
        response = model.generate_content(prompt)
        return {'response': response.text}
    except Exception as e:
        return {'response': f"Erreur : {str(e)}"}

if __name__ == '__main__':
    # Configuration pour l'environnement de développement
    app.run(debug=True, host='0.0.0.0', port=5001)