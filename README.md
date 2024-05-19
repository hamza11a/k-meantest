# k-meantest
We'll use the K-means clustering algorithm to segment customers based on their purchasing behavior.  Problem Statement : We have a  dataset of customers with their annual income and spending score at a shopping mall. Our task is to segment these customers into groups based on their similarities.
-------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate dummy data
data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 70, size=n_samples),
    'income': np.random.randint(20000, 150000, size=n_samples),
    'purchase_frequency': np.random.randint(1, 100, size=n_samples),
    'tenure': np.random.randint(1, 10, size=n_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())
# Select relevant features
features = df[['age', 'income', 'purchase_frequency', 'tenure']]

# Normalize the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Display normalized features
print(pd.DataFrame(normalized_features, columns=features.columns).head())
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(normalized_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
# Assuming k=3 from the Elbow Method
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(normalized_features)

# Add cluster labels to the original data
df['cluster'] = clusters

# Display the first few rows with cluster labels
print(df.head())
# Get centroids
centroids = kmeans.cluster_centers_

# Inverse transform the centroids to original scale
centroids = scaler.inverse_transform(centroids)

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=features.columns)
print(centroid_df)
# Get centroids
centroids = kmeans.cluster_centers_

# Inverse transform the centroids to original scale
centroids = scaler.inverse_transform(centroids)

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=features.columns)
print(centroid_df)
# Get centroids
centroids = kmeans.cluster_centers_

# Inverse transform the centroids to original scale
centroids = scaler.inverse_transform(centroids)

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=features.columns)
print(centroid_df)
