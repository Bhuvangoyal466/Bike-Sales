import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from textblob import TextBlob
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import folium
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
from data_preprocessing import load_and_clean_data
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Load data once
df = load_and_clean_data()


def geospatial_analysis(data):
    st.header("🌍 Geospatial Analysis of Bike Sales")
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    sales_map = folium.Map(location=map_center, zoom_start=5)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"Region: {row['Region']} | Sales: ${row['Sales']}",
        ).add_to(sales_map)

    st_folium(sales_map, width=700, height=500)


def customer_segmentation(data):
    st.header("👥 Customer Segmentation")
    k = st.slider("Select Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k)
    df["Cluster"] = kmeans.fit_predict(df[["Customer_Age", "Revenue"]])

    # Plot clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df["Customer_Age"], df["Revenue"], c=df["Cluster"], cmap="viridis"
    )
    plt.xlabel("Customer Age")
    plt.ylabel("Revenue")
    st.pyplot(fig)


# def anomaly_detection(data):
#     st.header("⚠️ Anomaly Detection")

#     # Apply Isolation Forest to detect anomalies in the 'Revenue' column
#     iso_forest = IsolationForest(contamination=0.05)
#     data["Anomaly"] = iso_forest.fit_predict(data[["Revenue"]])

#     # Visualize the results
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot normal points
#     ax.scatter(
#         data.index[data["Anomaly"] == 1],
#         data["Revenue"][data["Anomaly"] == 1],
#         color="blue",
#         label="Normal",
#         alpha=0.6,
#     )

#     # Plot anomalies
#     ax.scatter(
#         data.index[data["Anomaly"] == -1],
#         data["Revenue"][data["Anomaly"] == -1],
#         color="red",
#         label="Anomaly",
#         alpha=0.9,
#     )

#     ax.set_title("Anomaly Detection in Revenue", fontsize=16)
#     ax.set_xlabel("Index", fontsize=12)
#     ax.set_ylabel("Revenue", fontsize=12)
#     ax.legend()

#     st.pyplot(fig)

#     # Optionally show the anomalies in a table below the plot
#     st.write("Detected Anomalies:")
#     st.write(data[data["Anomaly"] == -1])


def anomaly_detection(data):
    st.header("⚠️ Anomaly Detection")

    # Apply Isolation Forest to detect anomalies in the 'Revenue' column
    iso_forest = IsolationForest(contamination=0.05)
    data["Anomaly"] = iso_forest.fit_predict(data[["Revenue"]])

    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot major normal points (sample a small number to reduce crowding)
    normal_data = data[data["Anomaly"] == 1]
    normal_sample = normal_data.sample(
        n=min(200, len(normal_data)), random_state=42
    )  # Limit to 200 points or fewer

    ax.scatter(
        normal_sample.index,
        normal_sample["Revenue"],
        color="lightblue",  # Softer color for normal points
        label="Normal",
        alpha=0.5,  # Softer transparency
    )

    # Plot anomalies (highlight all anomalies)
    anomaly_data = data[data["Anomaly"] == -1]
    ax.scatter(
        anomaly_data.index,
        anomaly_data["Revenue"],
        color="salmon",  # Softer red color for anomalies
        label="Anomaly",
        alpha=0.9,  # Higher transparency for anomalies
    )

    # Adding titles and labels
    ax.set_title("Anomaly Detection in Revenue", fontsize=16)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Revenue", fontsize=12)
    ax.legend()

    # Show the plot
    st.pyplot(fig)

    # Optionally show the anomalies in a table below the plot
    st.write("Detected Anomalies:")
    st.write(anomaly_data)


def predictive_analytics(data):
    st.header("🔮 Predictive Analytics")
    features = ["Feature1", "Feature2", "Feature3"]  # Adjust based on dataset
    target = "Sales"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = np.mean((y_test - predictions) ** 2)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write("Predictions vs Actuals:")
    st.write(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))


def interactive_filter(data):
    st.header("🔍 Interactive Filters and Insights")

    # Step 1: Filters
    st.sidebar.subheader("Apply Filters")
    country = st.sidebar.selectbox(
        "Select Country", options=df["Country"].unique(), index=0
    )
    state = st.sidebar.multiselect(
        "Select State(s)", options=df[df["Country"] == country]["State"].unique()
    )
    category = st.sidebar.multiselect(
        "Select Product Category", options=df["Product_Category"].unique()
    )

    # Step 2: Apply filters dynamically
    filtered_data = df[df["Country"] == country]
    if state:
        filtered_data = filtered_data[filtered_data["State"].isin(state)]
    if category:
        filtered_data = filtered_data[filtered_data["Product_Category"].isin(category)]

    # Step 3: Display Filtered Data
    st.subheader(f"Filtered Data: {len(filtered_data)} Rows")
    st.dataframe(filtered_data)

    # Step 4: Key Metrics
    total_revenue = filtered_data["Revenue"].sum()
    total_profit = filtered_data["Profit"].sum()
    total_orders = filtered_data["Order_Quantity"].sum()

    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
    col2.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    col3.metric(label="Total Orders", value=total_orders)

    # Step 5: Visualizations
    st.markdown("### Insights")

    # Revenue by State
    if state:
        fig_state_revenue = px.bar(
            filtered_data.groupby("State")["Revenue"].sum().reset_index(),
            x="State",
            y="Revenue",
            title="Revenue by State",
            text="Revenue",
        )
        st.plotly_chart(fig_state_revenue, use_container_width=True)

    # Revenue by Product Category
    if category:
        fig_category_revenue = px.pie(
            filtered_data.groupby("Product_Category")["Revenue"].sum().reset_index(),
            values="Revenue",
            names="Product_Category",
            title="Revenue Share by Category",
        )
        st.plotly_chart(fig_category_revenue, use_container_width=True)


def kpi_dashboard(data):
    st.header("📊 KPI Dashboard")

    # Total Sales - Bar Chart (Updated)
    total_sales = data["Revenue"].sum()
    other_metrics = (
        1000000 - total_sales
    )  # Placeholder for comparison (can be adjusted as needed)

    fig_sales = go.Figure(
        go.Bar(
            x=["Total Sales", "Other Metrics"],
            y=[total_sales, other_metrics],
            marker=dict(color=["#2ca02c", "#f4f4f4"]),
            text=[f"${total_sales:,.2f}", f"${other_metrics:,.2f}"],
            textposition="auto",
        )
    )
    fig_sales.update_layout(
        title="Total Sales Comparison",
        yaxis_title="Amount ($)",
        xaxis_title="Metrics",
        showlegend=False,
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    # Average Revenue per Customer - Bar Chart
    avg_revenue = data["Revenue"].mean()
    fig_avg_revenue = go.Figure(
        go.Bar(
            x=["Average Revenue per Customer"],
            y=[avg_revenue],
            text=[f"${avg_revenue:,.2f}"],
            textposition="auto",
            marker=dict(color="#00bfae"),
        )
    )
    fig_avg_revenue.update_layout(
        title="Average Revenue per Customer", yaxis_title="Revenue ($)"
    )
    st.plotly_chart(fig_avg_revenue, use_container_width=True)

    # Profit Margin - Donut Chart
    profit_margin = (data["Profit"].sum() / total_sales) * 100
    fig_profit_margin = go.Figure(
        go.Pie(
            labels=["Profit Margin", "Other Metrics"],
            values=[profit_margin, 100 - profit_margin],
            hole=0.4,
            marker=dict(colors=["#ff5733", "#f4f4f4"]),
        )
    )
    fig_profit_margin.update_layout(
        title="Profit Margin",
        annotations=[
            dict(
                text=f"{profit_margin:.2f}%",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig_profit_margin, use_container_width=True)

    # Additional Section: Total Profit vs Revenue Bar Chart
    total_profit = data["Profit"].sum()
    fig_profit_revenue = go.Figure(
        go.Bar(
            x=["Total Profit", "Total Revenue"],
            y=[total_profit, total_sales],
            text=[f"${total_profit:,.2f}", f"${total_sales:,.2f}"],
            textposition="auto",
            marker=dict(color=["#ff5733", "#2ca02c"]),
        )
    )
    fig_profit_revenue.update_layout(
        title="Total Profit vs Total Revenue", yaxis_title="Amount ($)"
    )
    st.plotly_chart(fig_profit_revenue, use_container_width=True)
