import numpy as np

sales_data = np.array([
    [1, 10, 50.0],
    [2, 5, 30.0],
    [3, 8, 20.0]
])

average_price = np.mean(sales_data[:, 2])

print("Average price of all products sold:", average_price)
