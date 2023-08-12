import matplotlib.pyplot as plt

# Sample data: Month and corresponding sales values
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [10000, 15000, 12000, 18000, 22000, 20000]

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Line plot
ax1.plot(months, sales, marker='o')
ax1.set_title('Monthly Sales (Line Plot)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales')

# Bar plot
ax2.bar(months, sales)
ax2.set_title('Monthly Sales (Bar Plot)')
ax2.set_xlabel('Month')
ax2.set_ylabel('Sales')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
