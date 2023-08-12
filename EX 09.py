import pandas as pd
data = {
    'property_id': [1, 2, 3, 4, 5],
    'location': ['City A', 'City B', 'City A', 'City C', 'City B'],
    'number_of_bedrooms': [3, 4, 3, 5, 4],
    'area_sq_feet': [1500, 2000, 1800, 2500, 1900],
    'listing_price': [250000, 350000, 280000, 450000, 320000]
}
property_data = pd.DataFrame(data)
average_price_per_location = property_data.groupby('location')['listing_price'].mean()
properties_with_more_than_four_bedrooms = property_data[property_data['number_of_bedrooms'] > 4]
num_properties_more_than_four_bedrooms = properties_with_more_than_four_bedrooms.shape[0]
property_with_largest_area = property_data[property_data['area_sq_feet'] == property_data['area_sq_feet'].max()] 
print("Average Listing Price per Location:\n", average_price_per_location)
print("\nNumber of Properties with More than Four Bedrooms:", num_properties_more_than_four_bedrooms)
print("\nProperty with the Largest Area:\n", property_with_largest_area)
