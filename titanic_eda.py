import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv - Copy.csv")

# -----------------------r
# BASIC INFO
# -----------------------
print("FIRST 5 ROWS")
print(df.head()) 

print("\nDATA INFO")
print(df.info())

# -----------------------
# MISSING VALUES (BEFORE CLEANING)
# -----------------------
print("\nMISSING VALUES (BEFORE):")
print(df.isnull().sum())

# -----------------------
# DATA CLEANING
# -----------------------
df["Age"] = df["Age"].fillna(df["Age"].mean())

# -----------------------
# MISSING VALUES (AFTER CLEANING)
# -----------------------
print("\nMISSING VALUES (AFTER):")
print(df.isnull().sum())

# -----------------------
# BASIC ANALYSIS
# -----------------------
print("\nSURVIVAL COUNT")
print(df["Survived"].value_counts())

print("\nSURVIVAL RATE (%)")
print(df["Survived"].value_counts(normalize=True) * 100)

# -----------------------
# VISUALIZATION
# -----------------------

# 1. Survival count
plt.figure(figsize=(8,5))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survived Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 2. Age group survival
plt.figure(figsize=(8,5))
df.groupby(pd.cut(df["Age"], 5))["Survived"].mean().plot(kind="bar")
plt.title("Age Group vs Survival")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.show()

# 3. Class survival rate
plt.figure(figsize=(8,5))
df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Class")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.show()

# 4. Age distribution
plt.figure(figsize=(8,5))
df["Age"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# -----------------------
# HEATMAP
# -----------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()