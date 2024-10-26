import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# Set up page configuration
st.set_page_config(page_title="Dashboard", layout="wide")

# Create a MySQL engine
engine = create_engine('mysql+mysqlconnector://root:minu2001@localhost/delivergate')

# Load data from MySQL
@st.cache_data
def load_data():
    with engine.connect() as conn:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        orders = pd.read_sql("SELECT * FROM orders", conn)
    return customers, orders

customers_df, orders_df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")

# Date range filter for order_date
start_date = st.sidebar.date_input("Start Date", value=orders_df['order_date'].min())
end_date = st.sidebar.date_input("End Date", value=orders_df['order_date'].max())
filtered_orders_df = orders_df[(orders_df['order_date'] >= pd.Timestamp(start_date)) & 
                               (orders_df['order_date'] <= pd.Timestamp(end_date))]

# Slider to filter by total amount spent by customers
# Calculate total amount spent per customer
customer_total_spent = filtered_orders_df.groupby('customer_id')['total_amount'].sum()

min_total_spent = st.sidebar.slider("Minimum Total Amount Spent by Customer", 
                                    min_value=float(customer_total_spent.min()), 
                                    max_value=float(customer_total_spent.max()), 
                                    value=1000.0, 
                                    step=100.0,
                                    help="Filter customers who have spent more than this amount.")

# Filter customers based on the total amount spent
filtered_customer_ids_by_spent = customer_total_spent[customer_total_spent >= min_total_spent].index
filtered_orders_df = filtered_orders_df[filtered_orders_df['customer_id'].isin(filtered_customer_ids_by_spent)]

# Dropdown to filter by number of orders
# Calculate the number of orders per customer
customer_order_counts = filtered_orders_df['customer_id'].value_counts()

min_order_count = st.sidebar.selectbox("Minimum Number of Orders", 
                                       options=[1, 5, 10, 15, 20], 
                                       index=1, 
                                       help="Filter customers with more than this number of orders.")

# Filter customers based on the selected minimum number of orders
filtered_customer_ids_by_orders = customer_order_counts[customer_order_counts > min_order_count].index
filtered_orders_df = filtered_orders_df[filtered_orders_df['customer_id'].isin(filtered_customer_ids_by_orders)]

# Main Dashboard

# Display filtered data in a table
st.header("Filtered Orders Data")
st.dataframe(filtered_orders_df)

# Bar chart of top 10 customers by revenue
st.subheader("Top 10 Customers by Revenue")
top_customers = filtered_orders_df.groupby('customer_id')['total_amount'].sum().nlargest(10)
st.bar_chart(top_customers)

# Line chart of total revenue over time (grouped by month)
st.subheader("Total Revenue Over Time")
revenue_over_time = filtered_orders_df.resample('M', on='order_date')['total_amount'].sum()
st.line_chart(revenue_over_time)

# Summary Metrics
st.header("Summary Metrics")
total_revenue = filtered_orders_df['total_amount'].sum()
unique_customers = filtered_orders_df['customer_id'].nunique()
total_orders = filtered_orders_df.shape[0]

st.metric("Total Revenue", f"${total_revenue:,.2f}")
st.metric("Number of Unique Customers", unique_customers)
st.metric("Total Number of Orders", total_orders)
