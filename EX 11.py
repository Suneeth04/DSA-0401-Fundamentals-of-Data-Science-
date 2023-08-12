import matplotlib.pyplot as plt

def plot_monthly_sales(months, sales):
    plt.bar(months, sales)
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Data')
    plt.grid(True)

months = [1, 2, 3, 4, 5, 6]
sales = [1000, 1200, 900, 1500, 1800, 1300]

plt.figure(figsize=(12, 6))

# Line plot with markers
plt.subplot(1, 3, 1)
plt.plot(months, sales, marker='o', linestyle='-',color='yellow')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Line Plot with Markers')
plt.grid(True)

# Scatter plot
plt.subplot(1, 3, 2)
plt.scatter(months, sales,color='blue')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Scatter Plot')
plt.grid(True)

# Bar plot
plt.subplot(1, 3, 3)
plot_monthly_sales(months, sales)

plt.tight_layout()
plt.show()
