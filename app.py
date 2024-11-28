import streamlit as st
import pandas as pd
from plots import *
from model_training import train_regression_model, train_classification_model
from PIL import Image
from features import *

# Set the page configuration
st.set_page_config(page_title="Bike Sales Analysis", page_icon=":bike:", layout="wide")

train_classification_model()
model, X_test, y_test, predictions = train_regression_model()


@st.cache_data
def load_combined_data():
    data = pd.read_csv("cleaned_sales_data.csv")
    required_columns = [
        "Date",
        "Customer_Age",
        "Country",
        "State",
        "Product_Category",
        "Order_Quantity",
        "Revenue",
        "Profit",
        "Unit_Cost",
        "Unit_Price",
    ]
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Missing column in dataset: {col}")
    return data


# Load and process data
# Load and process data
sales_data = load_combined_data()

customer_data = sales_data.loc[
    sales_data[["Customer_Age", "Age_Group", "Customer_Gender", "Country"]]
    .dropna()
    .index,
    ["Customer_Age", "Age_Group", "Customer_Gender", "Country"],
]

feedback_data = sales_data[["Product_Category", "Sub_Category", "Revenue"]].copy()
feedback_data["Review"] = feedback_data["Product_Category"].apply(
    lambda x: f"Great {x}!"
)

kpi_data = sales_data[
    ["Revenue", "Profit", "Order_Quantity", "Unit_Cost", "Unit_Price"]
]

forecasting_data = sales_data[["Date", "Revenue", "Order_Quantity"]]

filter_data = sales_data[
    ["Country", "State", "Product_Category", "Sub_Category", "Revenue"]
]

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Navigate to",
    [
        "Home",
        "Sales Analysis",
        "Customer Segmentation",
        "Prediction Results",
        "KPI Dashboard",
        "Additional Insights",
    ],
)

if page == "Home":
    st.title("🚴 Bike Sales Analysis Dashboard")
    st.write("Welcome to the Bike Sales Analysis Dashboard!")
    st.image(
        "screenshot.png",
        caption="Explore the world of bike sales!",
        use_column_width=True,
    )

elif page == "Sales Analysis":
    st.title("📈 Sales Analysis")
    plot_sales_trend()
    st.title("Interactive filters")
    interactive_filter(filter_data)

elif page == "Customer Segmentation":
    customer_segmentation(customer_data)

elif page == "Prediction Results":
    plot_prediction_results(y_test, predictions)


elif page == "KPI Dashboard":
    kpi_dashboard(kpi_data)

elif page == "Additional Insights":
    st.title("📊 Additional Insights")
    anomaly_detection(sales_data)
