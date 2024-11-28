import pandas as pd


def load_and_clean_data():
    # Load dataset
    df = pd.read_csv("data/Sales.csv")

    # Clean and preprocess the data
    df["Customer_Age"].fillna(df["Customer_Age"].median(), inplace=True)
    df["Customer_Gender"].fillna(df["Customer_Gender"].mode()[0], inplace=True)
    df["Country"] = df["Country"].str.title()
    df["State"] = df["State"].str.title()
    df["Product_Category"] = df["Product_Category"].str.title()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df
