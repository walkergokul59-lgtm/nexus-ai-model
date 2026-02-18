# ML Model Implementation Guide

This folder contains the Python implementation for the machine learning model described. It uses `scikit-learn` to classify log entries based on features extracted from various log types (Auth, Netflow, DNS, HTTP).

## Files

-   `modeling.py`: Contains the core logic for feature extraction, model definition, training, and inference.
-   `train_models.py`: A script to load the CSV data files in this directory and train the models.
-   `requirements.txt`: A list of Python dependencies required to run the code.
-   `synthetic_*.csv`: The data files used for training.

## Setup

1.  **Install Python**: Ensure you have Python installed (3.8 or higher is recommended).
2.  **Install Dependencies**: Open a terminal in this directory and run:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To train the models using the provided CSV files (`synthetic_auth_logs.csv`, etc.), run:

```bash
python train_models.py
```

This will:
1.  Read the CSV files.
2.  Extract features using the logic in `modeling.py`.
3.  Train a classifier for each log type.
4.  Save the trained model bundle to `model_bundle.joblib`.
5.  Print the performance metrics (Precision, Recall, etc.) for each model.

## Web Application

The project includes a web-based dashboard to visualize the ML model results.

### Running Locally

To run the web application on your machine:

1.  **FastAPI Server**:
    ```bash
    python -m uvicorn server:app --reload
    ```
2.  **Access the Dashboard**:
    Open `http://127.0.0.1:8000` in your browser.
    -   Upload personal or synthetic CSV logs (e.g., `synthetic_auth_logs.csv`).
    -   View the Sentinel AI Security Dashboard.

### Hosting on Vercel

The project is configured for easy deployment on **Vercel**:

1.  **Initialize Git**:
    ```bash
    git init
    git add .
    git commit -m "initial commit"
    ```
2.  **GitHub**: Create a new repository on GitHub and push your code.
3.  **Vercel Deployment**:
    -   Go to [vercel.com](https://vercel.com) and click **"New Project"**.
    -   Import your GitHub repository.
    -   Vercel will detect the `vercel.json` and Python environment.
    -   Click **"Deploy"**.

Once deployed, you can share the Vercel URL with your friends to show them the real-time AI security analysis!
