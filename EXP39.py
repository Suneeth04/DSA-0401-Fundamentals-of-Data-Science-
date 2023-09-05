import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
num_samples = 200
spending = np.random.randint(100, 1000, num_samples)
purchase_frequency = np.random.randint(1, 20, num_samples)
data = pd.DataFrame({'Spending': spending, 'PurchaseFrequency': purchase_frequency})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

num_clusters = 3 
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for cluster_num in range(num_clusters):
    cluster_data = data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Spending'], cluster_data['PurchaseFrequency'], color=colors[cluster_num], label=f'Cluster {cluster_num}')
    
plt.xlabel('Spending')
plt.ylabel('Purchase Frequency')
plt.title('Customer Clusters')
plt.legend()
plt.show()
