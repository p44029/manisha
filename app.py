import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso model
filename = 'lasso_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title("Lasso Regression Monthly Revenue Prediction")

# Get user input for the features
st.header("Enter Store Details:")
avg_order_value = st.number_input("Average Order Value", value=0.0)
total_orders = st.number_input("Total Orders", value=0.0)
customer_acquisition_cost = st.number_input("Customer Acquisition Cost", value=0.0)
avg_customer_lifetime_value = st.number_input("Average Customer Lifetime Value", value=0.0)
customer_churn_rate = st.number_input("Customer Churn Rate", value=0.0)
marketing_spend = st.number_input("Marketing Spend", value=0.0)
operational_cost = st.number_input("Operational Cost", value=0.0)


# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'avg_order_value': [avg_order_value],
    'total_orders': [total_orders],
    'customer_acquisition_cost': [customer_acquisition_cost],
    'avg_customer_lifetime_value': [avg_customer_lifetime_value],
    'customer_churn_rate': [customer_churn_rate],
    'marketing_spend': [marketing_spend],
    'operational_cost': [operational_cost]
})

# Make a prediction using the loaded model
if st.button("Predict Monthly Revenue"):
    prediction = loaded_model.predict(input_data)
    st.write("Predicted Monthly Revenue:", prediction[0])
