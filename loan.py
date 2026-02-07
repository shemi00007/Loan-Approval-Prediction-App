############################################################################
# Importing Libraries
############################################################################
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
############################################################################
# Streamlit Page Setup
############################################################################
st.set_page_config(page_title="Loan Approval Checking Portal", layout="wide")
st.title(" 💵💵 Loan Approval prediction Portal 💵💵 ")
st.caption("Machine Learning Classification Project using Loan Dataset from Bank")
############################################################################
# Data Importing / Loading (cached)
############################################################################
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
############################################################################
# Model Training
############################################################################
@st.cache_resource
def train_model(df:pd.DataFrame):
    target = "approved"

    drop_cols = [target]

    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    x = df.drop(columns=drop_cols)
    y = df[target]

    cat_cols = [c for c in["gender","city","employment_type","bank"] if c in x.columns]
    num_cols = [c for c in x.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numeric_transformer,num_cols),
            ("cat",categorical_transformer,cat_cols)
        ]
    )
    model = LogisticRegression(max_iter=2000)
    clf = Pipeline(steps=[
        ("preprocess",preprocessor),
        ("model",model)
    ])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2,random_state=42,stratify=y
    )
    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    metrics = {
        "accuracy" : float(accuracy_score(y_test,y_pred)),
        # precision --> when we predict approved, how often is it corect?
        "precision" : float(precision_score(y_test,y_pred,zero_division=0)),
        # recall --> out of all truly approved, how many did we catch? 
        "recall" : float(recall_score(y_test,y_pred,zero_division=0)),
        # f1 score --> balance between precision and recall
        "f1" : float(f1_score(y_test,y_pred,zero_division=0)),
        # confusion matrix --> it shows TP, FP, TN, FN in a 2*2 table
        "confusion matrix" : confusion_matrix(y_test,y_pred).tolist()
    }

    return clf, metrics, x_train.columns.tolist()
########################################################################
# Sidebar (1) Load Dataset
########################################################################

st.sidebar.header("(1) Load Dataset ")

csv_path = st.sidebar.text_input(
    "CSV Path",
    value= "loan_dataset.csv",
    help="Put the path to the dataset CSV. If you run from same folder, keep it as it is"
)

# Try Loadinf The Dataset
try:
    df = load_data(csv_path)

except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows ")

########################################################################
# Sidebar (2) Train Model
########################################################################
st.sidebar.header("(2) Train Model")
train_now = st.sidebar.button("Train / Re-Train")

if train_now:
    st.cache_resource.clear()

clf,metrics,feature_order = train_model(df)

########################################################################
# Main Layout
########################################################################

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

with colB:
    st.subheader("Model Metrics")

    st.write({
        "Accuracy" : round(metrics["accuracy"],4),
        "Precision" : round(metrics["precision"],4),
        "Recall" : round(metrics["recall"],4),
        "F1" : round(metrics["f1"],4),
    })
    cm = np.array(metrics["confusion matrix"])
    st.write("Confusion Matrix (row: actual [1,0], col: predicted[1,0])")

    st.dataframe(
        pd.DataFrame(cm, columns=["pred 1","pred 0"], index=["Actual 1","Actual 0"]),
        use_container_width=True
    )
st.divider()

########################################################################
# Trying A Prediction (UI Inputs)
########################################################################
st.subheader("Try a Prediction")

c1,c2,c3 = st.columns(3)

with c1:
    applicant_name = st.text_input("Applicant Name", value="Waseem Ahmed")
    gender = st.selectbox("Gender",["Male","Female"],index=0)
    age = st.slider("Age",21,60,30)
    city = st.selectbox("City", sorted(df['city'].unique().tolist()))
    employment_type = st.selectbox("Employment Type",sorted(df["employment_type"].unique().tolist()))

with c2:
    bank = st.selectbox("Bank",sorted(df["bank"].unique().tolist()))
    monthly_income_pkr = st.number_input("Monthly Income(PKR)",min_value=1500,max_value=500000,value=120000,step=1000)
    credit_score = st.slider("Credit Score",300,900,680)
    loan_amount_pkr = st.number_input("Loan Amount (PKR)",min_value=50000,max_value=3500000,value=800000,step=5000)

with c3:
    loan_tenure_months = st.selectbox("Tenure (Months)",[6,12,18,24,36,48,60],index=3)
    existing_loans = st.selectbox("Existing Loans",[0,1,2,3],index=0)
    default_history = st.selectbox("Default History",[0,1],format_func=lambda x: "No (0)"if x==0 else "Yes (1)",index=0)
    has_credit_card = st.selectbox("Has Credit Card",[0,1],format_func=lambda x: "No (0)"if x==0 else "Yes (1)",index=0)

########################################################################
# Building Model Input Rows
########################################################################
input_row = pd.DataFrame([{
    "gender"             : gender,
    "age"                : age,
    "city"               : city,
    "employment_type"    : employment_type,
    "bank"               : bank,
    "monthly_income_pkr" : monthly_income_pkr,
    "credit_score"       : credit_score,
    "loan_amount_pkr"    : loan_amount_pkr,
    "loan_tenure_months" : loan_tenure_months,
    "existing_loans"     : existing_loans,
    "default_history"    : default_history,
    "has_credit_card"    : has_credit_card
}])

input_row = input_row[feature_order]
########################################################################
# Prediction Button
########################################################################

if st.button("Predict Approval"):
    prob = float(clf.predict_proba(input_row)[:,1][0])

    pred = int(prob >=0.5)

    if pred == 1 :
        st.success(f" {applicant_name} : APPROVED (The loan repayment probability is: {prob:.2%})")
    else: 
        st.error(f" {applicant_name} :REJECTED (The loan repayment probability is: {prob:.2%})")