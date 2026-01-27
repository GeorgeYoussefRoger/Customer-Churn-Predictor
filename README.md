# 🏃 Customer Churn Predictor
A machine learning project that predicts customer churn using the Telco Customer Churn dataset. 
The project implements an end-to-end machine learning pipeline from data preprocessing and exploratory analysis to deep learning model training and deployment via a Streamlit web application.

## 🚀 Features
- End-to-end ML pipeline from raw data to production-ready web app
- Automated preprocessing and feature scaling for consistent inference
- EDA revealing key churn drivers such as tenure and monthly charges
- Deep learning classifier optimized for imbalanced churn prediction
- Interactive Streamlit app providing real-time churn probability and risk level

## 🧠 Methodology
1. **Data Preprocessing**
   - Removed irrelevant columns and handled missing values
   - Converted inconsistent data types

2. **Exploratory Data Analysis (EDA)**
   - Analyzed **Tenure vs. Churn** to understand retention patterns
   - Examined **Monthly Charges vs. Churn** to identify pricing-related churn behavior

3. **Feature Engineering**
   - Encoded categorical variables using binary and one-hot encoding
   - Applied stratified train-test splitting  
   - Scaled numerical features using StandardScaler (fit on training data only)

4. **Model Training**
   - Built a Sequential neural network using TensorFlow/Keras  
   - Architecture: `Input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)`
   - Optimizer: Adam  
   - Loss Function: Binary Crossentropy  
   - Handled class imbalance using class-weighted loss
   - Evaluated performance using confusion matrix and classification report

## 📂 Project Structure
```
Customer_Churn_Predictor/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Telco raw dataset
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── notebook/
│   ├── 01_data_preprocessing_eda.ipynb               # Data preprocessing + EDA
│   ├── 02_feature_engineering.ipynb                  # Feature engineering + Train-Test split
│   └── 03_model_training.ipynb                       # Model training + Evaluation
├── models/
│   ├── final_model.keras                             # Trained deep learning model
│   └── scaler.pkl                                    # Scaling for numerical features
├── ui/
│   └── app.py                                        # Streamlit app
├── requirements.txt                                  # Dependencies
├── README.md                                         
├── .gitignore
└── LICENSE
```

## 🧰 Technologies Used
- Language: Python (3.12)
- Data Analysis: `Pandas`
- Data Visualization: `Matplotlib` `Seaborn`
- Machine Learning and Deep Learning: `Scikit-Learn` `TensorFlow` `Keras`
- Model Persistence: `Joblib`
- Web App & Deployment: `Streamlit`

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
streamlit run ui/app.py
```

## 📂 Dataset
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📜 License
- This project is licensed under the MIT License.
- See the `LICENSE` file for more details.