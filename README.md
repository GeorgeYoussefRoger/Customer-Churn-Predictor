# 📉 Customer Churn Predictor
A Machine Learning–based Customer Churn Prediction system that estimates the probability of a customer leaving a telecom service provider. The project implements a full end-to-end classification workflow including data cleaning, exploratory data analysis (EDA), feature engineering, model training with imbalanced data handling, hyperparameter tuning, and deployment through an interactive Streamlit web application.

## 🚀 Features
- Predict customer churn probability based on demographic, service, and billing information
- End-to-end ML pipelines combining preprocessing, SMOTE, and classification
- Model comparison using cross-validated f1-score
- Hyperparameter tuning using GridSearchCV
- Single production-ready pipeline saved as a `.pkl` file
- Interactive Streamlit web interface for real-time predictions

## 🧠 Methodology
- Data Preprocessing & EDA
    - Removed irrelevant identifier columns
    - Cleaned numerical inconsistencies in TotalCharges
    - Performed exploratory data analysis:
        - Churn distribution
        - Tenure vs Churn
        - Monthly Charges vs Churn
    - Saved cleaned dataset for downstream modeling

- Feature Engineering
    - Encoded target variable: 
        - Churn: Yes → 1, No → 0
    - Standardized service-related categorical labels:
        - Replaced "No phone service" and "No internet service" with "No"
    - Performed stratified train–test split (80/20)
    - Saved raw train–test splits for reproducibility

- Model Training & Evaluation
    - Built full machine learning pipelines using `imblearn.Pipeline`:
        - Preprocessing with ColumnTransformer
        - Feature scaling for numerical variables
        - One-hot encoding for categorical features
        - SMOTE for class imbalance handling
        - Classification model
    - Trained and compared multiple classifiers:
        - Logistic Regression 
        - Random Forest
        - Gradient Boosting (best performing)
    - Evaluated models using:
        - 5-fold cross-validated f1-score (mean & standard deviation)
        - Precision, Recall on the test set
        - Selected the best baseline model based on f1-score
        - Saved artifact: `base_pipeline.pkl`

- Hyperparameter Tuning
    - Tuned Gradient Boosting using `GridSearchCV`
    - Used f1-score as the optimization metric
    - Evaluated tuned model on test data
    - Saved final optimized pipeline

- Deployment
    - Built an interactive Streamlit web application
    - User inputs customer details through structured UI sections
    - Outputs churn probability with risk interpretation
    - Includes basic input validation for billing consistency

## 📂 Project Structure
```
Customer-Churn-Predictor/
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
│ ├── clean_telco_churn.csv                 # Cleaned dataset
│ ├── X_train.csv                           # Training features
│ ├── X_test.csv                            # Test features
│ ├── y_train.csv                           # Training labels
│ └── y_test.csv                            # Test labels
├── notebooks/
│ ├── 01_data_preprocessing_eda.ipynb       # Cleaning and EDA
│ ├── 02_feature_engineering.ipynb          # Encoding and train-test split
│ ├── 03_model_training.ipynb               # Pipeline-based model comparison
│ └── 04_hyperparameter_tuning.ipynb        # Gradient Boosting tuning
├── models/
│ ├── base_pipeline.pkl                     # Best baseline pipeline
│ └── final_pipeline.pkl                    # Tuned final pipeline
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Project dependencies
├── README.md
├── .gitignore
└── LICENSE
```

## 🧰 Technologies Used
- Language: Python (3.12)
- Data Analysis: `Pandas`
- Data Visualization: `Matplotlib`, `Seaborn`
- Machine Learning: `Scikit-Learn`
- Imbalanced Learning: `Imbalanced-Learn` (SMOTE)
- Model Persistence: `Joblib`
- Web UI & Deployment: `Streamlit`

## 📦 Installation & Usage
1. Clone the repository
```
git clone https://github.com/GeorgeYoussefRoger/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the Streamlit app
```
streamlit run app.py
```

## 📂 Dataset
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contains customer demographics, subscribed services, billing details, and churn labels

## 📜 License
- This project is licensed under the MIT License.
- See the `LICENSE` file for more details.