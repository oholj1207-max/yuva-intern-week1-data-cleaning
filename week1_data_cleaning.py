import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# ---------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

# ---------------------------------------------------
# 2. INITIAL DATA EXPLORATION
# ---------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# ---------------------------------------------------
# 3. CHECK MISSING VALUES
# ---------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------
# 4. CHECK DUPLICATE VALUES
# ---------------------------------------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# ---------------------------------------------------
# 5. HANDLE MISSING VALUES
# ---------------------------------------------------

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has many missing values, so drop the column
df = df.drop(columns=["Cabin"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------
# 6. REMOVE DUPLICATES
# ---------------------------------------------------

df = df.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

# ---------------------------------------------------
# 7. CHECK DATA TYPES
# ---------------------------------------------------

print("\nData types:")
print(df.dtypes)

# ---------------------------------------------------
# 8. SAVE CLEANED DATASET
# ---------------------------------------------------

df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ---------------------------------------------------
# 9. FINAL DATASET INFORMATION
# ---------------------------------------------------

print("\nFinal dataset shape:", df.shape)

print("\nFinal dataset preview:")
print(df.head())

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!") 



# ---------------------------------------------------
# 10. OUTLIER DETECTION
# ---------------------------------------------------

print("\n--- OUTLIER DETECTION ---")

# Function to find outliers using IQR
def find_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"\nColumn: {column}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers)}")

# Check important numerical columns
find_outliers("Age")
find_outliers("Fare")

# ---------------------------------------------------
# 11. VISUALIZE OUTLIERS
# ---------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Fare"])
plt.title("Boxplot of Passenger Fare")
plt.xlabel("Fare")
plt.savefig("fare_boxplot.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Age"])
plt.title("Boxplot of Passenger Age")
plt.xlabel("Age")
plt.savefig("age_boxplot.png")
plt.show()

print("\nOutlier detection and visualization completed!")


print("\nFinal missing values after cleaning:")
print(df.isnull().sum())
