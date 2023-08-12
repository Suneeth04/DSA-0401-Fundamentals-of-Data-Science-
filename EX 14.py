import pandas as pd
def calculate_age_frequency_distribution(df):
    purchases_df = df[df['Purchase']]
    age_freq = purchases_df['Age'].value_counts().sort_index()
    return age_freq
if __name__ == "__main__":
    data = {
        'Customer_ID': [],
        'Age': [],
        'Purchase': []
    }
    n = int(input("Enter the number of customers: "))
    for i in range(n):
        customer_id = i + 1
        age = int(input(f"Enter the age of customer {customer_id}: "))
        purchase = input(f"Did customer {customer_id} make a purchase (yes/no)? ").lower() == 'yes'
        data['Customer_ID'].append(customer_id)
        data['Age'].append(age)
        data['Purchase'].append(purchase)

    df = pd.DataFrame(data)
    age_distribution = calculate_age_frequency_distribution(df)
    print("\nFrequency Distribution of Customer Ages:")
    for age, freq in age_distribution.items():
        print(f"{age}: {freq}")
