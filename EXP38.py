import numpy as np

def calculate_mean_temperature(temperatures):
    return np.mean(temperatures)

def calculate_std_deviation(temperatures):
    return np.std(temperatures)

def calculate_temperature_range(temperatures):
    return np.max(temperatures) - np.min(temperatures)

def calculate_most_consistent_std_deviation(temperatures):
    return np.min(np.std(temperatures, axis=0))

def main():

    dataset = {
        "City1": [25, 28, 27, 30, 31, 32, 30, 29, 28, 26, 25, 24],
        "City2": [15, 16, 16, 18, 20, 21, 19, 18, 17, 16, 15, 14],
        "City3": [10, 12, 13, 14, 15, 16, 16, 15, 14, 12, 10, 9]
    }

    city_temperatures = list(dataset.values())
    cities = list(dataset.keys())

    mean_temperatures = [calculate_mean_temperature(temps) for temps in city_temperatures]
    std_deviations = [calculate_std_deviation(temps) for temps in city_temperatures]
    temperature_ranges = [calculate_temperature_range(temps) for temps in city_temperatures]
    most_consistent_city_index = np.argmin(std_deviations)

    print("Mean Temperatures:")
    for i, city in enumerate(cities):
        print(f"{city}: {mean_temperatures[i]:.2f}°C")

    print("\nStandard Deviations:")
    for i, city in enumerate(cities):
        print(f"{city}: {std_deviations[i]:.2f}°C")

    max_range_index = np.argmax(temperature_ranges)
    most_consistent_city = cities[most_consistent_city_index]
    print(f"\nCity with the highest temperature range: {cities[max_range_index]}")
    print(f"City with the most consistent temperature: {most_consistent_city}")

if __name__ == "__main__":
    main()
