import matplotlib.pyplot as plt
def plot_monthly_data(months, temperatures, rainfall):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(months, temperatures, marker='o', linestyle='-',color='green')
    plt.xlabel('Month')
    plt.ylabel('Temperature (Celsius)')
    plt.title('Monthly Temperature Data')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(months, rainfall)
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.title('Monthly Rainfall Data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    temperatures = [25, 27, 28, 30, 32, 35, 36, 34, 33, 30, 28, 26]
    rainfall = [50, 45, 70, 80, 90, 110, 120, 100, 95, 85, 60, 55]
    plot_monthly_data(months, temperatures, rainfall)
