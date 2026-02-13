# 🔐 Login Anomaly Detection System

**A production-ready web application for detecting suspicious login attempts using machine learning.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Features](#-features)
3. [Installation](#-installation)
4. [How to Use](#-how-to-use)
5. [Understanding Results](#-understanding-results)
6. [API Documentation](#-api-documentation)
7. [Deployment](#-deployment)
8. [Troubleshooting](#-troubleshooting)
9. [IDE Warnings](#-expected-ide-warnings)

---

## 🚀 Quick Start

### Fastest Way to Run (Windows)

1. **Double-click `setup.bat`** (Only needed the first time)
2. **Double-click `run.bat`** (Starts the server)
3. **Open browser** to `http://localhost:5000`
4. **Click "Train Model"** button
5. **Start detecting anomalies!**

### Manual Method

```bash
# Install dependencies
pip install flask numpy pandas scikit-learn matplotlib seaborn joblib

# Run application
python app.py

# Open http://localhost:5000
```

---

## 🌟 Features

### 🌐 Web Interface

- **Modern UI**: Beautiful purple gradient design
- **Responsive**: Works on mobile, tablet, and desktop
- **Interactive**: Real-time feedback and smooth animations

### 🛠️ Functionality

- **Train Models**: Create custom ML models with adjustable parameters
- **Single Detection**: Analyze individual login attempts instantly
- **Batch Processing**: Upload CSV files to analyze thousands of records
- **Visualizations**: interactive charts for deep insights
- **Statistics**: Monitor model performance and anomaly rates
- **REST API**: Full API for integration with other systems

### 🔒 Security

- **Input Validation**: Protects against invalid data
- **Risk Assessment**: Classifies threats (Low, Medium, High, Critical)
- **Logging**: Comprehensive logs for auditing

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download the project**

   ```bash
   cd login_anomaly_detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   _Or use `setup.bat` on Windows_

3. **Run the application**
   ```bash
   python app.py
   ```
   _Or use `run.bat` on Windows_

---

## 🎯 How to Use

### 1️⃣ Train Your Model

1. Go to the **Train Model** tab (default).
2. Set **Samples** (e.g., 1000) and **Anomaly Ratio** (e.g., 5%).
3. Click **Train Model**.
4. Wait a few seconds for completion.

### 2️⃣ Detect Anomalies (Single)

1. Go to **Detect Anomalies** > **Single Login**.
2. Enter login details (Hour, Attempts, IP Frequency, etc.).
3. Click **Analyze Login**.
4. See if it's **Normal** or **Anomaly**.

### 3️⃣ Batch Analysis

1. Go to **Detect Anomalies** > **Batch Upload**.
2. Upload a CSV file with columns: `login_hour`, `login_attempts`, `ip_frequency`, `device_type`, `login_success`.
3. Click **Analyze Batch**.
4. Review results for all records.

### 4️⃣ Visualizations

1. Go to **Visualizations**.
2. Click **Generate Visualizations**.
3. View the dashboard with scatter plots and distributions.

### 5️⃣ Statistics

1. Go to **Statistics**.
2. Click **Refresh Statistics**.
3. View model metrics.

---

## 📊 Understanding Results

### Risk Levels

| Level        | Score         | Meaning           | Action                  |
| ------------ | ------------- | ----------------- | ----------------------- |
| **Critical** | < -0.1        | Highly suspicious | Immediate investigation |
| **High**     | -0.1 to -0.05 | Suspicious        | Investigate soon        |
| **Medium**   | -0.05 to 0    | Unusual           | Monitor                 |
| **Low**      | ≥ 0           | Normal            | No action               |

### Example Scenarios

**Normal Login**

- Business hours (9-17)
- 1-2 attempts
- Frequent IP
- Successful login

**Suspicious Login**

- Odd hours (2-5 AM)
- Many attempts (5+)
- Rare/New IP
- Failed login

---

## 🔌 API Documentation

Base URL: `http://localhost:5000/api`

### Endpoints

| Method | Endpoint     | Description                    |
| ------ | ------------ | ------------------------------ |
| POST   | `/train`     | Train a new model              |
| POST   | `/predict`   | Analyze single or batch logins |
| POST   | `/visualize` | Generate dashboard images      |
| GET    | `/stats`     | Get model statistics           |

**Example Request (Predict Single):**

```json
POST /api/predict
{
  "login_hour": 14,
  "login_attempts": 1,
  "ip_frequency": 50,
  "device_type": 1,
  "login_success": 1
}
```

---

## 🚀 Deployment

### Production with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 🔧 Troubleshooting

### App won't start?

- Ensure Python 3.8+ is installed.
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`.
- Check if port 5000 is in use.

### "Model not trained"?

- Go to the **Train Model** tab and click **Train Model** before predicting.

### CSV upload fails?

- Ensure columns match: `login_hour`, `login_attempts`, `ip_frequency`, `device_type`, `login_success`.
- Remove empty rows.

---

## ⚠️ Expected IDE Warnings

You might see type hint warnings in your IDE (like VS Code). **These are normal and safe to ignore.**

- **Why?** Libraries like Pandas and Scikit-learn have complex types that static analyzers struggle with.
- **Impact?** None. The code runs perfectly.
- **Action?** Ignore them or configure your IDE to be less strict.

---

## 📁 Project Structure

```
login_anomaly_detection/
├── app.py                # Flask web application
├── main.py               # CLI script
├── requirements.txt      # Dependencies
├── setup.bat             # Windows setup script
├── run.bat               # Windows run script
├── src/                  # Source code modules
│   ├── config.py
│   ├── logger.py
│   ├── data_generator.py
│   ├── preprocessor.py
│   ├── anomaly_detector.py
│   └── visualizer.py
├── static/               # CSS & JS
├── templates/            # HTML files
├── data/                 # Generated data
├── models/               # Saved models
├── outputs/              # Generated charts
└── logs/                 # Application logs
```

---

**Built with ❤️ using Flask & Scikit-learn**
