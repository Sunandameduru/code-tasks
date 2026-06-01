# Titanic Dataset - Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Titanic.csv")  # Replace with your file path

# 1. Basic Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# 3. Histograms for Numerical Features
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()

# 4. Boxplots for Numerical Features
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# 5. Correlation Matrix
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# 6. Pairplot
sns.pairplot(df[num_cols])
plt.show()

# 7. Survival Analysis
if 'Survived' in df.columns:

    plt.figure(figsize=(6,4))
    sns.countplot(x='Survived', data=df)
    plt.title("Survival Distribution")
    plt.show()

    if 'Sex' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='Sex', hue='Survived', data=df)
        plt.title("Survival by Gender")
        plt.show()

    if 'Pclass' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='Pclass', hue='Survived', data=df)
        plt.title("Survival by Passenger Class")
        plt.show()

print("\nEDA Completed Successfully!")
