import numpy as np
from scipy import stats

# Sample data for the new drug group (replace this with your actual data)
new_drug_data = []
new_drug = int(input("Enter the noof new drugs came in this year:"))
for i in range(0,new_drug):
    a = int(input("Enter the drug data:"))
    new_drug_data.append(a)
print(new_drug_data)
# Sample data for the placebo group (replace this with your actual data)
placebo_data = []
placebo = int(input("Enter the noof new placeholders came in this year:"))
for i in range(0,new_drug):
    b = int(input("Enter the placebo data:"))
    placebo_data.append(b)
print(placebo_data)

def calculate_confidence_interval(data):
    confidence_level = 0.95
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    sample_size = len(data)
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1) * std_dev / np.sqrt(sample_size)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound

if __name__ == "__main__":
    # Calculate confidence interval for the new drug group
    new_drug_lower, new_drug_upper = calculate_confidence_interval(new_drug_data)
    print(f"95% Confidence Interval for mean reduction in blood pressure (New Drug Group): [{new_drug_lower:.2f}, {new_drug_upper:.2f}]")

    # Calculate confidence interval for the placebo group
    placebo_lower, placebo_upper = calculate_confidence_interval(placebo_data)
    print(f"95% Confidence Interval for mean reduction in blood pressure (Placebo Group): [{placebo_lower:.2f}, {placebo_upper:.2f}]")

