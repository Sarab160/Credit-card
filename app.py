import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Delinquency Predictor", layout="wide")
st.title("💳 Credit Card Delinquency Prediction")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("credit_card.csv")

df = load_data()

# ---------------------------
# Feature Selection (YOUR CODE)
# ---------------------------
num_cols = [
    "Annual_Fees","Activation_30_Days","Customer_Acq_Cost",
    "current_year","Credit_Limit","Total_Revolving_Bal",
    "Total_Trans_Amt","Total_Trans_Vol",
    "Avg_Utilization_Ratio","Interest_Earned"
]

cat_cols = ["Card_Category","Week_Num","Use Chip","Qtr","Exp Type"]

X_num = df[num_cols]
y = df["Delinquent_Acc"]

ohe = OneHotEncoder(drop="first", sparse_output=False)
X_cat = ohe.fit_transform(df[cat_cols])
X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))

X = pd.concat([X_num, X_cat_df], axis=1)

# ---------------------------
# Oversampling + Split (YOUR CODE)
# ---------------------------
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling (needed for KNN)
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Model Training (AUTO)
# ---------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ---------------------------
# Accuracy & Metrics
# ---------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

f1_macro = f1_score(y_test, y_test_pred, average="macro")
recall_macro = recall_score(y_test, y_test_pred, average="macro")
precision_macro = precision_score(y_test, y_test_pred, average="macro")

st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Training Accuracy", f"{train_acc:.2%}")
col2.metric("Testing Accuracy", f"{test_acc:.2%}")
col3.metric("F1 Score (Macro)", f"{f1_macro:.2f}")

col4, col5 = st.columns(2)
col4.metric("Recall (Macro)", f"{recall_macro:.2f}")
col5.metric("Precision (Macro)", f"{precision_macro:.2f}")

st.subheader("📌 Classification Report")
st.text(classification_report(y_test, y_test_pred))

st.subheader("🧾 Confusion Matrix")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---------------------------
# User Input Prediction
# ---------------------------
st.subheader("🔮 Predict New Customer")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Annual_Fees = st.number_input("Annual Fees", 0.0)
        Activation_30_Days = st.number_input("Activation (30 Days)", 0.0)
        Customer_Acq_Cost = st.number_input("Customer Acquisition Cost", 0.0)
        current_year = st.number_input("Current Year", 0)
        Credit_Limit = st.number_input("Credit Limit", 0.0)

    with col2:
        Total_Revolving_Bal = st.number_input("Total Revolving Balance", 0.0)
        Total_Trans_Amt = st.number_input("Total Transaction Amount", 0.0)
        Total_Trans_Vol = st.number_input("Total Transaction Volume", 0.0)
        Avg_Utilization_Ratio = st.number_input("Avg Utilization Ratio", 0.0)
        Interest_Earned = st.number_input("Interest Earned", 0.0)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    user_num = pd.DataFrame([[
        Annual_Fees, Activation_30_Days, Customer_Acq_Cost,
        current_year, Credit_Limit, Total_Revolving_Bal,
        Total_Trans_Amt, Total_Trans_Vol,
        Avg_Utilization_Ratio, Interest_Earned
    ]], columns=num_cols)

    # categorical columns (zero-filled, same as training)
    user_cat = pd.DataFrame(
        [[0]*len(X_cat_df.columns)],
        columns=X_cat_df.columns
    )

    user_input = pd.concat([user_num, user_cat], axis=1)
    user_scaled = scaler.transform(user_input)

    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Delinquent Account")
    else:
        st.success("✅ Prediction: Non-Delinquent Account")
