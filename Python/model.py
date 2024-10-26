import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up page configuration
st.set_page_config(page_title="Model Comparison", layout="wide")

# Create a MySQL engine (replace with your MySQL credentials)
engine = create_engine('mysql+mysqlconnector://root:minu2001@localhost/delivergate')

# Load data from MySQL
@st.cache_data
def load_data():
    with engine.connect() as conn:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        orders = pd.read_sql("SELECT * FROM orders", conn)
    return customers, orders

customers_df, orders_df = load_data()

# 1. Prepare the data for the model
customer_orders = orders_df.groupby('customer_id').size()
customer_revenue = orders_df.groupby('customer_id')['total_amount'].sum()

data = pd.DataFrame({
    'total_orders': customer_orders,
    'total_revenue': customer_revenue
}).reset_index()

data['repeat_purchaser'] = (data['total_orders'] > 1).astype(int)

# Display Class Distribution
st.header("Data Analysis for Repeat Purchasers Prediction")
st.write("### Class Distribution")
st.write(data['repeat_purchaser'].value_counts().rename(index={0: "Non-repeat Customers", 1: "Repeat Customers"}))

# 2. Check if there’s enough data for training
if data['repeat_purchaser'].nunique() < 2:
    st.write("Not enough data to train the model. Please collect more data.")
else:
    # Prepare features and target variable
    X = data[['total_orders', 'total_revenue']]
    y = data['repeat_purchaser']

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Initialize dictionary to store results
    results = {}

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Train and evaluate each model
    for model_name, model in models.items():
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        # Store results
        results[model_name] = {
            "Accuracy": accuracy_scores.mean(),
            "Precision": precision_scores.mean(),
            "Recall": recall_scores.mean(),
            "F1 Score": f1_scores.mean()
        }

    # Display Results
    st.write("### Model Comparison Results")
    results_df = pd.DataFrame(results).T
    st.write(results_df)

    # Identify the best model
    best_model_name = results_df["F1 Score"].idxmax()
    st.write(f"### Best Model: {best_model_name}")
    st.write(f"**F1 Score**: {results_df.loc[best_model_name, 'F1 Score']:.2f}")
    
    # Train and evaluate the best model on a train-test split for final evaluation
    st.write("### Training the Best Model on a Train-Test Split for Final Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Final evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    st.write(f"**Final Accuracy**: {accuracy:.2f}")
    st.write(f"**Final Precision**: {precision:.2f}")
    st.write(f"**Final Recall**: {recall:.2f}")
    st.write(f"**Final F1 Score**: {f1:.2f}")
