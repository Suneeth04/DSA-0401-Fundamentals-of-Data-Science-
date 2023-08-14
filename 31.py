import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('customer_data.csv')
data = data.drop(['customer_id'], axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

data['cluster'] = cluster_labels
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=data.columns[:-1])

for i in range(n_clusters):
    print(f"Cluster {i}:\n{cluster_centers_df.iloc[i]}\n")

