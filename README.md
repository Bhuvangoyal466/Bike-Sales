# Bike-Sales
Bike Sales Analysis - Data Analysis Web App

The Bike Sales Analysis is a Python-based web application developed using Streamlit to analyze bike sales data. This interactive dashboard leverages data visualization and machine learning models to provide insights into sales trends, customer behavior, and key performance indicators (KPIs). It also includes prediction capabilities using regression and classification models.

# Features =>

# Streamlit-Powered Interactive Dashboard:
  Intuitive navigation sidebar for seamless access to multiple analytical pages.
  Dynamically rendered charts, KPIs, and prediction results.
  
# Data Loading and Processing:
  Data is loaded from a cleaned CSV file (cleaned_sales_data.csv).
  Validation ensures all necessary columns are present to avoid runtime errors.
  Processed subsets of data for specific tasks like customer segmentation, KPI calculation, and sales forecasting.
  
# Data Analysis Modules:

  # Sales Analysis:
    Visualizes trends in sales revenue, order quantity, and profit across time or geography.
    Includes interactive filters for dynamic exploration of sales by country, state, and product categories.
  # Customer Segmentation:
    Categorizes customers by age, gender, and region to identify key demographics.
  # KPI Dashboard:
    Displays key performance indicators such as revenue, profit, unit cost, and order quantity.
 #  Anomaly Detection:
    Identifies outliers or unusual patterns in sales data for deeper business insights.

# Machine Learning Models:
  
  # Regression Model: 
    Predicts future sales and revenue trends using historical data.
  # Classification Model: 
    Segments customers or identifies high-revenue-generating categories.
  # Prediction Results: 
    Visualized to compare model predictions with actual outcomes.

# Custom Features and Visualizations:
  Sales trend plotting (plot_sales_trend()).
  Interactive filtering for granular insights (interactive_filter()).
  Forecasting and anomaly detection for actionable intelligence.
  
# User Engagement:
  The home page includes an overview of the dashboard's purpose and an image showcasing bike sales.
  Clear navigation options to explore different analytical aspects.
  
# Implementation Details

# Navigation Sidebar:
  Built with st.sidebar.radio, providing easy access to pages like Home, Sales Analysis, Customer Segmentation, and Prediction Results.
  
# Data Handling:  
  Data is read using pandas and cached with @st.cache_data for efficient loading.
  Data subsets are created for different analytical tasks, ensuring modular and reusable code.
  
# Model Integration:
  Machine learning models (train_regression_model, train_classification_model) are trained during app initialization.
  Prediction results and test data are visualized interactively.
  
# UI and Visualizations:
  Dynamic, visually appealing charts for KPIs, trends, and segmentation.
  Additional insights are supported by anomaly detection.
  
# Workflow => 

# Home Page:
  Welcomes the user with an introduction and a header image.
  
# Sales Analysis:
  Visualizes sales trends and supports interactive filtering to explore specific regions, categories, or sub-categories.
  
# Regression Models:
  Displays regression model performance and predictions for sales and revenue.
  
# Customer Segmentation:
  Provides insights into customer demographics using predefined segmentation features.
  
# Prediction Results:
  Compares actual vs. predicted results for regression and classification tasks.

# KPI Dashboard:
  Summarizes essential business metrics like revenue, profit, and cost.
  
# Additional Insights:
  Highlights anomalies(outliers) in the data for further exploration.
  
This dashboard simplifies data analysis and visualization for stakeholders, making data-driven decisions more accessible and actionable in the context of bike sales.

# Power BI dashboard

A Power BI dashboard has been created for advanced data visualization and is showcased on the Home Page of the web app.
The dashboard provides a detailed overview of sales trends, customer demographics, and revenue distribution.

# Improved Visual Appeal:
  The embedded screenshot highlights the advanced insights available in Power BI, adding a professional touch to the web app.

# Complementary Insights:
  Users can compare Streamlit-generated analytics with Power BI’s polished visualizations, offering both in-depth technical analysis and executive-ready reports.

# Accessible Storytelling:
  The combination of Streamlit and Power BI ensures the app caters to both technical users and business stakeholders.
  
This integration makes the platform versatile and engaging for a broader audience.
![image](https://github.com/user-attachments/assets/840ff670-ee8c-4957-8a2e-3835d70fae3b)
![Screenshot 2024-11-28 103736](https://github.com/user-attachments/assets/fd0f0606-300b-4900-b965-ae39dfa4fd82)
![Screenshot 2024-11-28 103945](https://github.com/user-attachments/assets/1d6958e2-d22e-4ffc-bed8-6226d7513329)
![Screenshot 2024-11-28 103955](https://github.com/user-attachments/assets/95dfcade-d6eb-44a2-82e1-ca9ed4161b36)
![Screenshot 2024-11-28 104123](https://github.com/user-attachments/assets/915d5878-586f-483e-8857-dbdc9f44f8c3)
![Screenshot 2024-11-28 104136](https://github.com/user-attachments/assets/d2a31e2d-b879-4e2a-9a35-af5de283b4ef)




