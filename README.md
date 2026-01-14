# 🏃 Customer Churn Predictor
A machine learning project that predicts customer churn using the Telco Customer Churn dataset. 
The project implements an end-to-end pipeline from data preprocessing and EDA to deep learning model training and deployment via a Streamlit web application.

## 🚀 Features
- End-to-end ML pipeline from raw data to production-ready web app
- Automated preprocessing and feature scaling for consistent inference
- EDA revealing key churn drivers such as tenure and monthly charges
- Deep learning classifier optimized for imbalanced churn prediction
- Interactive Streamlit app providing real-time churn probability and risk level

## 🧠 Methodology
1. **Data Preprocessing**
   - Removed irrelevant columns and handled missing values
   - Encoded categorical variables using One-Hot Encoding
   - Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to improve neural network convergence

2. **Exploratory Data Analysis (EDA)**
   - Analyzed **Tenure vs. Churn** to understand retention patterns
   - Examined **Monthly Charges vs. Churn** to identify pricing-related churn behavior

3. **Model Training**
   - Built a Sequential neural network using TensorFlow/Keras  
   - Architecture: `Input → Dense(26, ReLU) → Dense(13, ReLU) → Dense(1, Sigmoid)`
   - Optimizer: Adam  
   - Loss Function: Binary Crossentropy  
   - Evaluated performance using confusion matrix and classification report

4. **Handling Class Imbalance**
   - Observed imbalance between churned and non-churned customers
   - Applied stratified train-test splitting
   - Used class-weighted loss to penalize false negatives (missed churn cases)

5. **Deployment**
   - Exported the trained model in `.keras` format
   - Persisted feature scaler using `joblib`
   - Deployed the model via a Streamlit web application that:
     - Accepts customer inputs
     - Outputs churn probability
     - Classifies risk as Low / Moderate / High

## 📂 Project Structure
```
Customer_Churn_Predictor/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Original dataset
├── notebook/
│   └── churn_predictor.ipynb                         # EDA, preprocessing, and model training
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