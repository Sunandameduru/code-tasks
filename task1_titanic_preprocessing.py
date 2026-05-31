# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("Titanic.csv")  # Replace with your dataset path

# Display Basic Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
# Fill numerical columns with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill categorical columns with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column if too many missing values
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Convert Categorical Data to Numerical
label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nEncoded Dataset:")
print(df.head())

# Standardize Numerical Features
scaler = StandardScaler()

numerical_cols = ['Age', 'Fare']

for col in numerical_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

print("\nData After Standardization:")
print(df.head())

# Visualize Outliers using Boxplots
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Remove Outliers using IQR Method
for col in ['Age', 'Fare']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# Save Cleaned Dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nData Cleaning & Preprocessing Completed Successfully!")
