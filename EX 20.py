import numpy as np
from scipy import stats
def perform_ttest(data_A, data_B):
    t_statistic, p_value = stats.ttest_ind(data_A, data_B)
    return t_statistic, p_value
if __name__ == "__main__":
    design_A_data = [0.1, 0.2, 0.15, 0.25, 0.18]
    design_B_data = [0.12, 0.22, 0.14, 0.20, 0.17]
    t_statistic, p_value = perform_ttest(design_A_data, design_B_data)
    alpha = 0.05
    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < alpha:
        print("There is a statistically significant difference in the mean conversion rates between website design A and website design B.")
    else:
        print("There is no statistically significant difference in the mean conversion rates between website design A and website design B.")
