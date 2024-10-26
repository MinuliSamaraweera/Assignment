import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

# Set up page configuration
st.set_page_config(page_title="Data Analysis", layout="wide")

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
# Calculate total orders and total revenue per customer
customer_orders = orders_df.groupby('customer_id').size()
customer_revenue = orders_df.groupby('customer_id')['total_amount'].sum()

# Create a DataFrame for machine learning
data = pd.DataFrame({
    'total_orders': customer_orders,
    'total_revenue': customer_revenue
}).reset_index()

# Define repeat purchaser: 1 if total_orders > 1, else 0
data['repeat_purchaser'] = (data['total_orders'] > 1).astype(int)

# Display Class Distribution
st.header("Data Analysis for Repeat Purchasers Prediction")
st.write("### Class Distribution")
st.write(data['repeat_purchaser'].value_counts().rename(index={0: "Non-repeat Customers", 1: "Repeat Customers"}))

# 2. Check if there’s enough data for training
if data['repeat_purchaser'].nunique() < 2:
    st.write("Not enough data to train the model. Please collect more data.")
else:
    # 3. Train-test split
    X = data[['total_orders', 'total_revenue']]
    y = data['repeat_purchaser']
    
    # Stratified split to maintain the proportion of classes in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation for more reliable accuracy estimate
    st.write("### Cross-Validation Results")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    st.write(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Display Results in Streamlit
    st.write("### Model Performance on Test Set")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    # Feature Coefficients
    st.write("### Feature Coefficients (Logistic Regression)")
    coefficients = pd.DataFrame(model.coef_, columns=['total_orders', 'total_revenue'])
    st.write(coefficients)

    # User-friendly Message Based on Performance
    if accuracy > 0.8:
        st.success("The model has a good accuracy in predicting repeat purchasers.")
    elif accuracy > 0.6:
        st.warning("The model accuracy is moderate. Consider adding more data or features for improvement.")
    else:
        st.error("The model accuracy is low. You might need more data or additional features.")
