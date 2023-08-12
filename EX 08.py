import pandas as pd
data = {
    "Product A": 10,
    "Product B": 70,
    "Product C": 140,
    "Product D": 121,
    "Product E": 85,
    "Product F": 60,
}
file = pd.DataFrame((data.items()), columns=["Product", "Quantity Sold"])
file_sorted = file.sort_values(by="Quantity Sold", ascending=False)
top = file_sorted.head(5)

print("Top 5 products sold the most in the past month:")
print(top)
