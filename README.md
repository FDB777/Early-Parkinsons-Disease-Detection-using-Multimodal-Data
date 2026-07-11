# Early Parkinson's Disease Detection Using Multimodal Data

A research and educational Flask application that combines voice recordings, handwriting/spiral images, and MRI images to estimate Parkinson's disease risk using machine-learning models.

> **Important:** This project is not a medical device and must not be used to diagnose, treat, or make clinical decisions about any person. Predictions require clinical validation and qualified medical interpretation.

## Features

- Accepts one or more input modalities: voice (`.wav`), handwriting (`.jpg`, `.jpeg`, `.png`), and MRI (`.jpg`, `.jpeg`, `.png`, `.nii`, `.nii.gz`).
- Extracts modality-specific features and produces individual predictions.
- Combines available modality predictions using a weighted ensemble.
- Uses trained CNN and LightGBM-style model artefacts for handwriting, MRI, and voice workflows.
- Provides user sign-up, login, prediction history, and generated report views through a Flask web interface.
- Includes explainability-oriented output for supported voice and MRI prediction flows.

## How it works

```text
Voice / handwriting / MRI input
            -> preprocessing and feature extraction
            -> modality-specific ML models
            -> weighted ensemble prediction
            -> report and visual explanations
```

The current ensemble weights are configured as voice 20%, handwriting 30%, and MRI 50%. When not all modalities are provided, the app normalises the weights of the available predictions.

## Technology stack

| Technology | Purpose |
|---|---|
| Python and Flask | Web application and prediction workflow |
| TensorFlow/Keras | CNN model loading and inference |
| scikit-learn and Joblib | Traditional ML model artefacts and serialisation |
| OpenCV, NumPy, scikit-image, NiBabel | Image and MRI processing |
| Parselmouth, NeuroKit2, Nolds, pyrpde | Voice-signal and nonlinear feature extraction |
| MySQL | User accounts, uploaded-input metadata, predictions, and report data |
| HTML, CSS, JavaScript | Server-rendered user interface |

## Repository structure

```text
Parkinsons_disease_model/
├── Files/parkinsons/        # Flask application, templates, static assets, requirements
├── project/
│   ├── voice/               # Voice data, feature extraction, and model artefacts
│   ├── handwriting/         # Handwriting preprocessing, training code, and model artefacts
│   └── mri/                 # MRI training code, normalisation data, and model artefacts
└── test_data/               # Example multimodal test inputs
```

## Run locally

### Requirements

- Python 3.10 or later is recommended.
- MySQL server with the application schema available.
- Sufficient RAM and compatible system libraries for TensorFlow and image/audio dependencies.

### Setup

```bash
git clone https://github.com/FDB777/Early-Parkinsons-Disease-Detection-using-Multimodal-Data.git
cd Early-Parkinsons-Disease-Detection-using-Multimodal-Data/Parkinsons_disease_model/Files/parkinsons
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Configure the database connection as environment variables. Do not put real credentials in source code or commit them to GitHub.

```text
DB_HOST=your-mysql-host
DB_PORT=3306
DB_USER=your-mysql-user
DB_PASSWORD=your-secure-password
DB_NAME=parkinsons
```

Before any public deployment, update the Flask configuration to read its secret key from an environment variable such as `SECRET_KEY`, rather than keeping it in source code.

Start the application:

```bash
python app.py
```

Then open the local address printed by Flask, commonly `http://127.0.0.1:5000`.

## Deployment notes

This is a full Flask and machine-learning application, not a static website. It needs:

- a Python runtime capable of installing TensorFlow and the image/audio dependencies;
- access to the sibling `project/` directory containing model files;
- a managed MySQL database; and
- persistent object/file storage for user uploads and generated reports.

For these reasons, a Python application host such as Render, Railway, or Google Cloud Run is more appropriate than static hosting. Do not use Vercel as the only host for the current architecture.

## Data and security

- Never upload or commit real patient data, personally identifiable health data, passwords, API keys, or database dumps.
- Store production secrets only in the hosting provider's environment-variable settings.
- Use encrypted storage, access control, retention policies, and informed consent before handling any real health-related data.
- Treat included sample data as development/testing material only.

## Limitations

- The model outputs are research predictions, not a clinically validated diagnosis.
- Performance can vary with input quality, data distribution, hardware, and preprocessing.
- Model fairness, calibration, clinical evaluation, privacy compliance, and regulatory requirements must be addressed before any real-world healthcare use.
- Uploads and reports require production-grade storage and access controls before public deployment.
