import pandas as pd
data = {
    'customer_id': [1, 2, 1, 3, 2],
    'order_date': ['2023-01-01', '02-01-2023', '2023-01-03', '2023-01-01', '2023-01-02'],
    'product_name': ['Product A', 'Product B', 'Product A', 'Product C', 'Product B'],
    'order_quantity': [2, 3, 1, 4, 2]
}
order_data = pd.DataFrame(data)
total_orders_per_customer = order_data.groupby('customer_id')['order_date'].count()
average_quantity_per_product = order_data.groupby('product_name')['order_quantity'].mean()
earliest_order_date = order_data['order_date'].min()
latest_order_date = order_data['order_date'].max()
print("Total orders per customer:")
print(total_orders_per_customer)
print("\nAverage order quantity per product:")
print(average_quantity_per_product)
print("\nEarliest order date:", earliest_order_date)
print("Latest order date:", latest_order_date)
