import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv("houseprice.csv")

# Display first few rows
print(dataset.head())

# Dataset shape
print("Dataset shape:", dataset.shape)

# Check for missing values
print("Missing values:\n", dataset.isnull().sum())

# Summary statistics
print("Dataset description:\n", dataset.describe())

# Pairplot for visual analysis
sns.pairplot(data=dataset)
plt.show()

# Correlation heatmap
sns.heatmap(data=dataset.corr(), annot=True)
plt.show()

# Features and target
x = dataset.iloc[:, :-1]  # All columns except 'Price'
y = dataset["Price"]      # Target column

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(x_train, y_train)

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Sqft_living:", dataset['Living Area'].min(), "to", dataset['Living Area'].max())
print("Bedrooms:", dataset['Bedrooms'].min(), "to", dataset['Bedrooms'].max())
print("Bathrooms:", dataset['Bathrooms'].min(), "to", dataset['Bathrooms'].max())
print("Lot Size:", dataset['Lot Size'].min(), "to", dataset['Lot Size'].max())

# Accuracy on test set
accuracy = model.score(x_test, y_test) * 100
print("Accuracy of model (R² score):", accuracy, "%")

# Predictions on test set
y_pred = model.predict(x_test)

# User input for prediction
Sqft_living = float(input("Enter size in sq_ft: "))
Bedrooms = int(input("Enter Bedrooms: "))
Bathrooms = int(input("Enter Bathrooms: "))
Lot_Size = float(input("Enter Lot Size: "))

# Predict using the model
Price_predicted = model.predict([[Sqft_living, Bedrooms, Bathrooms, Lot_Size]])
print(f"Predicted Price for ${Sqft_living} area of {Bedrooms} Bedrooms and {Bathrooms} Bathrooms and Lot Size {Lot_Size}: ₹{Price_predicted[0]:.2f}")

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2 * 100:.2f}%")

# Plot: Actual vs Predicted (Test set)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7, label="Actual data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Prediction line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Testing Data)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Actual vs Predicted (Training set)
y_train_pred = model.predict(x_train)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='green', edgecolor='k', alpha=0.7, label="Training Data")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Training Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
