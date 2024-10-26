# Internship Data Engineer Technical Test

# Streamlit Dashboard for Customer Orders

This Streamlit app is a dashboard for visualizing and filtering customer orders from a MySQL database. It includes filters for order date, minimum total amount spent, and minimum number of orders, and displays metrics, bar charts, and line charts to summarize customer spending and order patterns.

## Features

- Date range filter for `order_date`
- Slider to filter customers based on total spending
- Dropdown to filter by minimum number of orders
- Summary metrics for total revenue, unique customers, and number of orders
- Bar chart for top 10 customers by revenue
- Line chart of total revenue over time

## Requirements

- Python 3.7 or later
- Streamlit
- MySQL database with tables for `customers` and `orders`
- Additional dependencies listed in `requirements.txt`

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/MinuliSamaraweera/Technical-Test
    cd Technical-Test
    ```

2. **Install Dependencies**:
    Install the necessary Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up the MySQL Database**:
   - Ensure you have a MySQL server running.
   - Create a database called `delivergate`.
   - Create the required tables (`customers` and `orders`) and insert data. Here are the schemas:

   ```sql
   CREATE TABLE customers (
       customer_id INT PRIMARY KEY,
       customer_name VARCHAR(100),
       email VARCHAR(100)
   );

   CREATE TABLE orders (
       order_id INT PRIMARY KEY,
       customer_id INT,
       total_amount DECIMAL(10, 2),
       order_date DATE,
       FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
   );

