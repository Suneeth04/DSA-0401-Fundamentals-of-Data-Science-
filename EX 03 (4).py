import numpy as np
house_data = np.genfromtxt("C:/Users/Nithin Sanjay/Downloads/Housing.csv", delimiter=',', skip_header=1)
bedrooms_column = house_data[:, 0]
condition = bedrooms_column > 4 
filtered_data = house_data[condition] 
average_sale_price = np.mean(filtered_data[:, 2])  
print("Average sale price of houses with more than four bedrooms:", average_sale_price)
