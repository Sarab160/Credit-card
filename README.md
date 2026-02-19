# 💳 Credit Card Delinquency Prediction App

A machine learning web application built with Streamlit that predicts whether a credit card account is likely to become delinquent using a K-Nearest Neighbors (KNN) classifier. The app includes automated model training, performance evaluation, and real-time prediction for new customers.

---

## 🚀 Features

- Automatic data preprocessing and feature encoding
- Class imbalance handling with Random Over Sampling
- Feature scaling for optimal KNN performance
- Model evaluation with accuracy, precision, recall, and F1-score
- Classification report and confusion matrix visualization
- Interactive form for real-time delinquency prediction

---

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn

---

## 📂 Project Structure

credit-delinquency-app/
│
├── app.py
├── credit_card.csv
├── requirements.txt
└── README.md

---

## 📊 Dataset

The dataset contains financial and transactional customer information such as:

- Credit limit and annual fees
- Transaction amount and volume
- Utilization ratio
- Interest earned
- Card category and usage behavior
- Delinquency status (target variable)

---

## ⚙️ Installation

1. Clone the repository:

git clone https://github.com/your-username/credit-delinquency-app.git

2. Navigate to the project folder:

cd credit-delinquency-app

3. Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run the App

streamlit run app.py

The app will open in your browser.

---

## 📈 Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Encoding: One-hot encoding for categorical features
- Balancing: RandomOverSampler
- Scaling: StandardScaler
- Metrics: Accuracy, Precision, Recall, F1-score
- Visualization: Confusion matrix heatmap

---

## 🧪 How to Use

1. Launch the app
2. View model performance metrics
3. Enter customer financial details
4. Get instant delinquency prediction

---

## 📌 Future Improvements

- Hyperparameter tuning
- Try additional ML models
- Model persistence and deployment
- Enhanced UI and dashboards

---

## 👨‍💻 Author

Your Name

---

## 📄 License

This project is open-source and free to use.
