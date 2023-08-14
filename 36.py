import pandas as pd
import numpy as np

def load_stock_data(filename):
    data = pd.read_csv(filename)
    return data

def calculate_variability(closing_prices):
    price_std = np.std(closing_prices)
    price_range = np.max(closing_prices) - np.min(closing_prices)
    return price_std, price_range

def provide_insights(variability_std):
    if variability_std < 10:
        return "The stock's prices have relatively low variability."
    elif variability_std < 50:
        return "The stock's prices have moderate variability."
    else:
        return "The stock's prices have high variability."

def main():
    filename = 'stock_data.csv'
    stock_data = load_stock_data(filename)
    
    closing_prices = stock_data['closing_price']
    
    price_std, price_range = calculate_variability(closing_prices)
    
    insights = provide_insights(price_std)
    
    print(f"Price Standard Deviation: {price_std:.2f}")
    print(f"Price Range: {price_range:.2f}")
    print(insights)

if __name__ == "__main__":
    main()
