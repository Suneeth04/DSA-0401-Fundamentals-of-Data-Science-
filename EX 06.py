item_prices = [2.5, 1.0, 3.0, 4.0]
item_quantities = [3, 5, 2, 1]
discount_rate = 10
tax_rate = 8

total_cost_before_discount = sum(p * q for p, q in zip(item_prices, item_quantities))
discount_amount = (discount_rate / 100) * total_cost_before_discount
total_cost_after_discount = total_cost_before_discount - discount_amount
tax_amount = (tax_rate / 100) * total_cost_after_discount
final_total_cost = total_cost_after_discount + tax_amount

print("Total cost before discount and tax:", total_cost_before_discount)
print("Total discount amount:", discount_amount)
print("Total cost after discount:", total_cost_after_discount)
print("Total tax amount:", tax_amount)
print("Final total cost including tax:", final_total_cost)
