# 🎓 Student GPA Prediction System - ML Microservice



## 📋 Project Overview

An end-to-end MLOps system that predicts student GPA using machine learning, deployed as a production-ready  with complete CI/CD automation. This service integrates seamlessly with a full-stack educational platform to provide real-time academic performance predictions.

## 🚀 Key Features

- **ML Model**: Random Forest Regressor trained on student academic data
- **Production-Ready API**: Flask-based REST API with health monitoring
- **Automated CI/CD**: Multi-stage GitHub Actions workflows (dev → staging → production)
- **Containerized Deployment**: Docker + Railway cloud platform
- **Version Control**: MLflow for model tracking, DVC for data versioning
- **Comprehensive Testing**: Unit, integration, and end-to-end test suites

## 🏗️ Architecture
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend   │─────▶│ ML Container│
│   (React)   │      │  (FastAPI)  │      │   (Flask)   │
└─────────────┘      └─────────────┘      └─────────────┘
                            │                     │
                            ▼                     ▼
                     ┌─────────────┐      ┌─────────────┐
                     │  PostgreSQL │      │   MLflow    │
                     │  (Database) │      │  (DagsHub)  │
                     └─────────────┘      └─────────────┘
```

## 🛠️ Technology Stack

### Machine Learning
- **Framework**: scikit-learn 1.5.2
- **Model Tracking**: MLflow + DagsHub
- **Data Versioning**: DVC with cloud storage
- **Features**: 12 engineered features including study hours, academic year, major, demographics

### DevOps & Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: GitHub Actions (4-stage CI/CD pipeline)
- **Cloud Platform**: Railway (production deployment)
- **Registry**: DockerHub
- **Testing**: pytest with 9 comprehensive tests

### API & Backend
- **Framework**: Flask with Gunicorn (production server)
- **Preprocessing**: Custom feature engineering pipeline
- **Endpoints**: `/predict`, `/health`
- **Response Format**: JSON with prediction confidence

## 📊 Model Performance

- **Algorithm**: Random Forest Regressor
- **Features**: 12 student attributes
- **Target**: GPA (0.0 - 4.0 scale)
- **Deployment**: Automated fetching of latest model from MLflow registry

## 🔄 CI/CD Pipeline

### Workflow 1: Pull Request Validation
- Trigger: PR to `dev` branch
- Actions: Build app + Run integration tests

### Workflow 2: Dev to Staging Sync
- Trigger: Push to `dev`
- Actions: Auto-merge to `staging` branch

### Workflow 3: Staging Validation
- Trigger: Push to `staging`
- Actions: Run all tests (unit + integration + e2e) → Auto-merge to `main`

### Workflow 4: Production Deployment
- Trigger: Push to `main`
- Actions: Build Docker image → Push to DockerHub → Deploy to Railway

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.10+
Docker
Git
```

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/Reemfad/student-gpa-prediction.git
cd student-gpa-prediction
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Create .env file
MLFLOW_TRACKING_URI=https://dagshub.com/reemfad51/student-gpa-prediction.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
```

4. **Run the application**
```bash
python src/app.py
```

The API will be available at `http://localhost:5000`

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

### Docker Deployment
```bash
# Build image
docker build -t gpa-predictor .

# Run container
docker run -p 5000:5000 \
  -e MLFLOW_TRACKING_URI=your_uri \
  -e MLFLOW_TRACKING_USERNAME=your_username \
  -e MLFLOW_TRACKING_PASSWORD=your_token \
  gpa-predictor
```

## 📡 API Usage

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1"
}
```

### Predict GPA
```bash
POST /predict
Content-Type: application/json

{
  "student_id": 12345,
  "uni_name": "University Name",
  "major": "Computer Science",
  "academic_year": 3,
  "study_hours": 7.5,
  "disability": false,
  "athleticstatus": "Active",
  "countryoforigin": "Country",
  "countryofresidence": "Country",
  "dropout": false
}
```

**Response:**
```json
{
  "student_id": 12345,
  "predicted_gpa": 3.45,
  "model_version": "1",
  "features_used": 12
}
```

## 📂 Project Structure
```
student-gpa-prediction/
├── .github/workflows/       # CI/CD pipeline definitions
│   ├── pr-to-dev.yml
│   ├── sync-dev-to-staging.yml
│   ├── staging-to-production.yml
│   └── deploy_to_production.yml
├── data/
│   ├── raw/                 # Original datasets (DVC tracked)
│   └── processed/           # Processed data (DVC tracked)
├── models/
│   └── label_encoders.pkl   # Feature encoding artifacts
├── notebooks/
│   └── train_model.ipynb    # Model training & experimentation
├── src/
│   ├── app.py              # Flask API application
│   └── preprocessing.py    # Feature engineering pipeline
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔐 Security & Best Practices

- ✅ Secrets managed via GitHub Secrets
- ✅ Environment variables for sensitive data
- ✅ Production-grade server (Gunicorn)
- ✅ Health monitoring endpoints
- ✅ Comprehensive error handling
- ✅ Input validation and sanitization

## 📈 Monitoring & Maintenance

- **Model Versioning**: MLflow tracks all model versions with metadata
- **Data Versioning**: DVC ensures reproducible datasets
- **Automated Testing**: 9 tests covering critical functionality
- **Deployment Health**: `/health` endpoint for monitoring
- **Logging**: Structured logging for debugging

## 🤝 Integration

This ML microservice integrates with:
- **Frontend**: React-based student portal
- **Backend**: FastAPI application handling business logic
- **Database**: PostgreSQL storing student profiles and predictions

## 🔮 Future Enhancements

- [ ] Model retraining pipeline automation
- [ ] A/B testing for model versions
- [ ] Advanced feature engineering
- [ ] Performance metrics dashboard
- [ ] Batch prediction endpoints
- [ ] Model explainability (SHAP values)

## 👥 Team

**ML & DevOps Engineer**: Responsible for ML model development, containerization, CI/CD pipeline, and cloud deployment

**Backend Developer**: API integration and database management

**Frontend Developer**: User interface and experience

## 📄 License

This project is part of a Machine Learning in Production course assignment.

## 🙏 Acknowledgments

- **MLflow** for experiment tracking
- **DagsHub** for MLOps platform
- **Railway** for cloud deployment
- **DVC** for data version control

---

**Live Demo**: [https://frontend-production-7f19.up.railway.app/dashboard]

**Documentation**: Full API documentation available at `/docs` endpoint