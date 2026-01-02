# Project Overview

This is a **Machine Learning Deployment Workshop** project. Its primary goal is to demonstrate the end-to-end process of training a machine learning model and deploying it as a web service.

The project focuses on predicting **Customer Churn** for a telecommunications company. It utilizes **Scikit-learn** for model training and **FastAPI** for serving the model via a REST API. Dependency management is handled by **uv**.

# Key Components

## Source Code
*   **`train.py`**: The training pipeline. It fetches the Telco Customer Churn dataset, preprocesses the data (encoding categorical variables), trains a Logistic Regression model, and serializes the trained pipeline to `model.pkl`.
*   **`predict.py`**: The inference service. It loads the `model.pkl` and uses **FastAPI** to expose a `/predict` endpoint. It uses **Pydantic** for rigorous input data validation.
*   **`marketing.py`**: A client-side script that simulates a marketing system. It sends a sample customer profile to the running prediction service and decides whether to send a promotional email based on the churn probability.
*   **`ping.py`**: A minimal FastAPI application providing a `/ping` endpoint, likely for connectivity testing or health checks.

## Configuration & Artifacts
*   **`model.pkl`**: The trained Scikit-learn pipeline (created by running `train.py`).
*   **`pyproject.toml` & `uv.lock`**: Definition of project dependencies and lock file, managed by `uv`.
*   **`Dockerfile`**: (Currently empty) Intended for containerizing the application.

# Building and Running

## Prerequisites
*   Python 3.14+
*   `uv` (Universal Python Package Installer)

## Setup
Install the project dependencies:
```bash
uv sync
```

## Workflow

1.  **Train the Model**:
    Execute the training script to fetch data and generate `model.pkl`.
    ```bash
    uv run train.py
    ```

2.  **Run the Prediction Service**:
    Start the FastAPI server. This will serve the model on `http://0.0.0.0:9696`.
    ```bash
    uv run predict.py
    ```

3.  **Test the Service**:
    In a separate terminal window, run the marketing script to send a request to the server.
    ```bash
    uv run marketing.py
    ```

## Other Commands
*   **Run Ping Service**: `uv run ping.py` (Starts a simple ping server on port 9696).

# Development Conventions

*   **Frameworks**: `FastAPI` for APIs, `Scikit-learn` for ML.
*   **Data Validation**: `Pydantic` models are strictly used for API request/response validation (see `Customer` and `PredictResponse` classes in `predict.py`).
*   **Data Processing**: Scikit-learn `Pipeline` and `ColumnTransformer` are used to encapsulate preprocessing steps (OneHotEncoding) with the model, ensuring consistency between training and inference.
*   **Dependency Management**: All dependencies are managed via `uv` in `pyproject.toml`.
