import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import load_and_clean_data


def train_regression_model():
    df = load_and_clean_data()

    # Features and target for regression
    X = df[["Customer_Age", "Order_Quantity", "Unit_Cost"]]
    y = df["Revenue"]

    # Train-test split (75-25 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


def train_classification_model():
    df = load_and_clean_data()

    # Features and target for classification
    X = df[["Customer_Age", "Order_Quantity", "Unit_Cost"]]
    y = (df["Profit"] > 0).astype(int)  # Binary classification: Profit > 0

    # Train-test split (75-25 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train the decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)
    return classifier, X_test, y_test, y_pred
