import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('house_data.csv')
selected_feature = 'size'
target_variable = 'price'

plt.scatter(data[selected_feature], data[target_variable])
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.title(f'Bivariate Analysis: {selected_feature} vs. {target_variable}')
plt.show()
X = data[[selected_feature]]
y = data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.title(f'Actual vs. Predicted: {selected_feature} vs. {target_variable}')
plt.legend()
plt.show()
