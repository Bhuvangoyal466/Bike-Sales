import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
from data_preprocessing import load_and_clean_data
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Load data once
df = load_and_clean_data()


def plot_sales_trend():
    df["Year"] = df["Date"].dt.year
    sales_per_year = df.groupby("Year")["Revenue"].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Year", y="Revenue", data=sales_per_year, marker="o", color="red")
    plt.title("Bike Sales Trend Over the Years", fontsize=16)
    plt.ylabel("Total Revenue", fontsize=14)
    plt.xlabel("Year", fontsize=14)
    st.pyplot(plt)


def plot_age_order_quantity(data):
    # Bin the ages into ranges
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = [
        "0-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-100",
    ]
    data["Age_Range"] = pd.cut(
        data["Customer_Age"], bins=bins, labels=labels, right=False
    )

    # Calculate average order quantity by age range
    age_order_data = data.groupby("Age_Range")["Order_Quantity"].mean().reset_index()

    fig, ax1 = plt.subplots()

    # Bar plot
    sns.barplot(
        data=age_order_data,
        x="Age_Range",
        y="Order_Quantity",
        palette="Blues",
        ax=ax1,
    )
    ax1.set_xlabel("Customer Age Range")
    ax1.set_ylabel("Average Order Quantity", color="b")
    ax1.set_title("Order Quantity by Customer Age Range")

    # Line plot
    ax2 = ax1.twinx()
    sns.lineplot(
        data=age_order_data,
        x="Age_Range",
        y="Order_Quantity",
        color="r",
        marker="o",
        ax=ax2,
    )
    ax2.set_ylabel("Order Quantity Trend", color="r")

    st.pyplot(fig)


def plot_profit_by_region():
    profit_per_country = df.groupby("Country")["Profit"].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Profit", y="Country", data=profit_per_country, palette="Reds")
    plt.title("Profit Distribution by Country", fontsize=16)
    plt.xlabel("Total Profit", fontsize=14)
    plt.ylabel("Country", fontsize=14)
    st.pyplot(plt)


def plot_revenue_by_category():
    revenue_by_category = df.groupby("Product_Category")["Revenue"].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Revenue", y="Product_Category", data=revenue_by_category, palette="dark:red"
    )
    plt.title("Revenue by Product Category", fontsize=16)
    plt.xlabel("Total Revenue", fontsize=14)
    plt.ylabel("Product Category", fontsize=14)
    st.pyplot(plt)


def plot_monthly_sales():
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly_sales = df.groupby("Month")["Revenue"].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Month", y="Revenue", data=monthly_sales, marker="o", color="red")
    plt.title("Monthly Bike Sales Trend", fontsize=16)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Total Revenue", fontsize=14)
    st.pyplot(plt)


def plot_customer_age_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Customer_Age"], bins=30, color="red", kde=True)
    plt.title("Distribution of Customer Age", fontsize=16)
    plt.xlabel("Customer Age", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    st.pyplot(plt)


def plot_profit_margins():
    profit_margin = (df["Profit"] / df["Revenue"]) * 100
    df["Profit_Margin"] = profit_margin

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Profit_Margin", y="Product_Category", data=df, palette="Reds")
    plt.title("Profit Margins by Product Category", fontsize=16)
    plt.xlabel("Profit Margin (%)", fontsize=14)
    plt.ylabel("Product Category", fontsize=14)
    st.pyplot(plt)


def plot_prediction_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color="red")
    plt.title("Actual vs Predicted Revenue", fontsize=16)
    plt.xlabel("Actual Revenue", fontsize=14)
    plt.ylabel("Predicted Revenue", fontsize=14)
    plt.xlim(0, 20000)
    plt.ylim(0, 10000)
    plt.xticks(range(0, 20001, 2000))
    plt.yticks(range(0, 10001, 2000))
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", lw=2
    )
    st.pyplot(plt)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# used for evaluating the performance of a classification model
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)


def plot_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="blue", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)

