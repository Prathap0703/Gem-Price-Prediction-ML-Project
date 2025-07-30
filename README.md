# 💎 Gemstone Price Prediction

A machine learning project that predicts gemstone prices based on various attributes like carat, cut, color, clarity, etc. This project includes an end-to-end pipeline for data ingestion, transformation, model training, evaluation, and a Flask web application for real-time prediction.

---

##  Project Structure

Gemstone Price Prediction/
├── app.py # Flask app entry point
├── src/ # Core ML components and pipeline
│ ├── components/ # Data ingestion, transformation, model trainer
│ ├── pipeline/ # Training and prediction pipelines
│ ├── exception.py # Custom exception class
│ ├── logger.py # Logging utility
│ └── utils.py # Reusable utility functions
├── notebook/ # Jupyter Notebooks for EDA and experiments
├── templates/ # HTML files for frontend
├── static/ # CSS or JS files
├── artifacts/ # Generated artifacts (models, data splits)
├── logs/ # Log files
├── requirements.txt # Python dependencies
├── setup.py # For packaging the project
├── .gitignore # Files/folders to ignore in git
├── .github/workflows/ # GitHub Actions for CI
│ └── main.yml
└── README.md # Project overview

## Project Steps
This project follows a modular, pipeline-driven workflow for building and deploying a machine learning model to predict gemstone prices.

# 1. Data Ingestion
The raw dataset (gemstone.csv) is read using Pandas.
The data is split into training and testing sets.
Both splits are saved as .csv files in the artifacts/ directory for reproducibility.
This logic is encapsulated inside a custom class-based pipeline (DataIngestion).

# 2. Data Transformation
A custom ColumnTransformer pipeline is built to preprocess both numeric and categorical features.
Numeric features:
Missing values are handled using SimpleImputer(strategy='median').
Features are standardized using StandardScaler.
Categorical features:
Missing values are filled using SimpleImputer(strategy='most_frequent').
Categories are encoded using Ordinal Encoding.
Features are scaled using StandardScaler.
The entire preprocessing pipeline is saved as a .pkl file for later use during prediction.

# 3. Model Training
Several regression models are trained with hyperparameter tuning and evaluated on the processed dataset.
Random Forest was the best performing model
The trained final model is saved as .pkl file.

# 4. Prediction Pipeline
The PredictionPipeline class handles:
Input data transformation using the saved preprocessor.
Model loading and prediction.
Input data (from the web app) is converted to a DataFrame internally.
The pipeline returns the predicted gemstone price to the user.

# 5. Flask Web Application
A lightweight Flask app is created with a user-friendly interface.
Users can input gemstone features via a web form.
On submission, the backend loads the pre-trained model and returns a predicted price.
Frontend is built with basic HTML/CSS inside the templates/ and static/ folders.