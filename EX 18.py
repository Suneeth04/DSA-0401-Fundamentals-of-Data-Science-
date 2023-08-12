import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

age_data = []
body_fat_data = []

persons = int(input("Enter the noof persons:"))
for i in range(0,persons):
    b = int(input("Enter the age:"))
    age_data.append(b)
    c = float(input("Enter the fat percentage:"))
    body_fat_data.append(c)

data = pd.DataFrame({'Age': age_data, '%Fat': body_fat_data})
mean_age = data['Age'].mean()
median_age = data['Age'].median()
std_age = data['Age'].std()

mean_fat = data['%Fat'].mean()
median_fat = data['%Fat'].median()
std_fat = data['%Fat'].std()

print("Age - Mean:", mean_age)
print("Age - Median:", median_age)
print("Age - Standard Deviation:", std_age)

print("%Fat - Mean:", mean_fat)
print("%Fat - Median:", median_fat)
print("%Fat - Standard Deviation:", std_fat)

plt.figure(figsize=(8, 6))
sns.boxplot(data=data)
plt.title("Boxplots for Age and %Fat")
plt.xlabel("Variables")
plt.ylabel("Values")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='%Fat')
plt.title("Scatter Plot of Age vs %Fat")
plt.xlabel("Age")
plt.ylabel("%Fat")
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(data['Age'], plot=plt)
plt.title("Q-Q Plot of Age")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(data['%Fat'], plot=plt)
plt.title("Q-Q Plot of %Fat")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()
