# 📊 Customer Churn Predictor

A machine learning project that predicts the likelihood of customer churn using the Telco Customer Churn dataset. This project demonstrates ML and DL skills — including data preprocessing, exploratory data analysis (EDA), model training, evaluation and deployment through a Streamlit app.

## 🚀 Features

- Full machine learning pipeline from raw data to web deployment
- Data cleaning, encoding, and standardization
- Exploratory Data Analysis (EDA) with churn-based visualizations
- Deep learning model with TensorFlow/Keras
- Interactive Streamlit app for real-time churn prediction

## 🧠 Methodology

🧹 Data Preprocessing

- Removed irrelevant columns and handled missing values
- Converted categorical variables to numeric via One-Hot Encoding
- Scaled numerical features (tenure, MonthlyCharges, TotalCharges) using StandardScaler

📊 Exploratory Data Analysis (EDA)

- Visualized churn patterns across features such as:
  - Tenure vs. Churn — to identify retention trends
  - Monthly Charges vs. Churn — to study spending-related churn behavior

🤖 Model Training

- Built a Sequential Neural Network using TensorFlow/Keras:
  - Input → Dense(26, ReLU) → Dense(13, ReLU) → Dense(1, Sigmoid)
  - Optimized with Adam and Binary Crossentropy
  - Evaluated with confusion matrix, and classification report

⚙️ Deployment

- Exported the final model (.keras format)
- Developed a Streamlit UI for user-friendly prediction

## 📂 Project Structure

```
Customer_Churn_Predictor/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Original dataset
├── notebook/
│   ├── churn_predictor.ipynb                         # Preprocessing + Neural network model
├── models/
│   ├── final_model.keras                             # Trained deep learning model
│   ├── scaler.pkl                                    # Scaling for numerical features
├── ui/
│   ├── app.py                                        # Streamlit app
├── requirements.txt                                  # Dependencies
├── README.md                                         # Documentation
├── .gitignore
├── LICENSE
```

## 🧰 Technologies Used

- Python — pandas, scikit-learn, tensorflow, keras, matplotlib, seaborn
- Web App — Streamlit

## 📦 Installation & Usage

1️⃣ Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

2️⃣ Install dependencies

```
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app

```
streamlit run ui/app.py
```

## 📂 Dataset

- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📜 License

- This project is licensed under the MIT License.
- See the `LICENSE` file for more details.
