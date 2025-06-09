import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="House Price Prediction App", layout="wide")

st.title("🏠 House Price Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("houseprice.csv")
    return df

dataset = load_data()

# Show dataset info
st.subheader("Dataset Preview")
st.write(dataset.head())

st.write(f"Dataset shape: {dataset.shape}")

st.subheader("Missing Values")
st.write(dataset.isnull().sum())

st.subheader("Dataset Description")
st.write(dataset.describe())

# Pairplot & heatmap can be slow, so add toggle
if st.checkbox("Show Pairplot (might take some time)"):
    st.subheader("Pairplot")
    sns_plot = sns.pairplot(dataset)
    st.pyplot(sns_plot)

if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dataset.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Prepare data
X = dataset.iloc[:, :-1]  # All but last column (Price)
y = dataset["Price"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Show model info
st.subheader("Model Parameters")
st.write(f"Intercept: {model.intercept_}")
st.write(f"Coefficients: {model.coef_}")

st.write(f"Living Area range: {dataset['Living Area'].min()} to {dataset['Living Area'].max()}")
st.write(f"Bedrooms range: {dataset['Bedrooms'].min()} to {dataset['Bedrooms'].max()}")
st.write(f"Bathrooms range: {dataset['Bathrooms'].min()} to {dataset['Bathrooms'].max()}")
st.write(f"Lot Size range: {dataset['Lot Size'].min()} to {dataset['Lot Size'].max()}")

# Model accuracy
accuracy = model.score(x_test, y_test) * 100
st.write(f"Model accuracy (R² score): {accuracy:.2f}%")

# User inputs for prediction
st.subheader("Make a Prediction")
sqft = st.number_input("Enter Living Area (sqft):", min_value=0.0, format="%.2f")
bedrooms = st.number_input("Enter Bedrooms:", min_value=0, step=1)
bathrooms = st.number_input("Enter Bathrooms:", min_value=0, step=1)
lot_size = st.number_input("Enter Lot Size:", min_value=0.0, format="%.2f")

if st.button("Predict Price"):
    input_data = np.array([[sqft, bedrooms, bathrooms, lot_size]])
    price_pred = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹{price_pred:,.2f}")

# Predictions on test set
y_pred = model.predict(x_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics on Test Set")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2 * 100:.2f}%")

# Plot Actual vs Predicted (Test)
st.subheader("Actual vs Predicted Price (Test Set)")
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7, label="Actual vs Predicted")
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted Price (Testing Data)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Plot Actual vs Predicted (Train)
st.subheader("Actual vs Predicted Price (Training Set)")
y_train_pred = model.predict(x_train)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(y_train, y_train_pred, color='green', edgecolor='k', alpha=0.7, label="Actual vs Predicted")
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label="Perfect Prediction")
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual vs Predicted Price (Training Data)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
