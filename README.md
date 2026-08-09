# PathSense 🚦

**PathSense** is an advanced AI-powered pipeline and microservices architecture designed to predict high-risk pedestrian accidents and issue real-time alerts. It integrates machine learning, computer vision, and real-time sensor ingestion to create a comprehensive safety net for smart cities and autonomous systems.

---

https://github.com/user-attachments/assets/0d1ced27-b342-4ca6-b046-178bb9e054f2

## 🚀 Key Features

* **Realistic ML Pipeline**: Features a machine learning pipeline (v2 & v3) using LightGBM. It utilizes Platt Scaling (`CalibratedClassifierCV`) to output true probability scores, achieving realistic **~71% accuracy** and **0.62 ROC-AUC** on real-world datasets with highly engineered features.
* **Microservices Architecture**: Built with modern, asynchronous FastAPI microservices:
  * `sensor_ingest`: Handles real-time telemetry and IoT sensor data.
  * `vision`: Processes camera feeds and visual hazard detection.
  * `alert_service`: Manages PostgreSQL database logging and Twilio SMS/email notifications.
* **Production-Grade Security**: 
  * 🔒 **Authentication**: Secured via `X-API-Key` headers across all microservices.
  * 🛡️ **CORS**: Strict `ALLOWED_ORIGINS` configuration preventing unauthorized frontend access.
  * 🔑 **Secrets Management**: Fail-fast startup validation for database URLs and Twilio credentials.
* **CI/CD Integration**: Automated GitHub Actions workflows that run `pytest` and validate model accuracy on every push.
* **Interactive Frontend**: A beautiful web dashboard equipped with Leaflet maps to visualize sensor data, track camera feeds, and display real-time alert notifications.

## 📂 Project Structure

```text
PathSense/
├── .github/workflows/    # CI/CD pipelines
├── api/                  # API Gateway / Routing
├── dataset/              # Real-world training data (e.g. pedestrian_accidents.csv)
├── frontend/             # Dashboard UI (HTML, CSS, Leaflet JS)
├── infra/                # Infrastructure & Docker (postgres-init.sql, etc.)
├── ml/                   # Machine Learning Models
│   ├── artifacts/        # Serialized models, encoders, and metrics reports
│   ├── train_v2.py       # Advanced v2 pipeline (LightGBM on real data)
│   ├── train_v3.py       # Unified schema v3 pipeline (Simulation / Dynamic)
│   └── predictor_v2.py   # Dynamic model loader & inference engine
├── monitoring/           # Grafana dashboards & Prometheus configs
├── services/             # FastAPI Microservices
│   ├── alert_service/
│   ├── sensor_ingest/
│   └── vision/
└── docker-compose.yml    # Local multi-container orchestration
```

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.9+
* Docker & Docker Compose (optional, for infra)
* PostgreSQL

### 1. Environment Variables
Create a `.env` file in the root directory and configure your secure variables:
```env
# Security
API_KEY=your_secure_api_key
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/pathsense

# Twilio (Alerts)
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+123456789
```

### 2. Install Dependencies
Install the required packages for the ML pipeline and services:
```bash
pip install -r requirements.txt
pip install -r requirements-services.txt
pip install -r requirements-vision.txt
```

### 3. Run the Infrastructure
Spin up the PostgreSQL database and monitoring stack:
```bash
docker-compose up -d
```

### 4. Train the ML Model
To generate the latest artifacts on your real-world dataset:
```bash
python ml/train_v2.py
```
*(This will save calibrated models to `ml/artifacts/` and update `model_report_v2.json`)*

### 5. Start the Services
Launch the individual FastAPI microservices (using uvicorn):
```bash
uvicorn services.sensor_ingest.main:app --port 8001 --reload
uvicorn services.vision.main:app --port 8002 --reload
uvicorn services.alert_service.main:app --port 8003 --reload
```

## 📈 Model Performance
The current production model (`lgbm_v2_calibrated.pkl`) has been fine-tuned for high recall on severe accidents:
- **Accuracy**: ~71%
- **ROC-AUC**: ~0.62
- **Class 1 (High-Risk) Recall**: 100%

*Continuous monitoring is available via the included Grafana dashboard.*
