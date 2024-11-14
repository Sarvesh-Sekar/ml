import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data_size = 100
glucose = np.random.normal(120, 15, data_size)
bmi = np.random.normal(28, 4, data_size)
age = np.random.randint(20, 70, data_size)

# Create DataFrame
data = pd.DataFrame({
    'Glucose': glucose,
    'BMI': bmi,
    'Age': age
})

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Define number of clusters
k = 3

# K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)
data['KMeans_Cluster'] = kmeans_labels

# Expectation-Maximization (Gaussian Mixture Model) clustering
em_model = GaussianMixture(n_components=k, random_state=42)
em_labels = em_model.fit_predict(scaled_features)
data['EM_Cluster'] = em_labels

# Calculate silhouette scores
kmeans_silhouette = silhouette_score(scaled_features, kmeans_labels)
em_silhouette = silhouette_score(scaled_features, em_labels)

print(f'Silhouette Score for K-Means: {kmeans_silhouette:.4f}')
print(f'Silhouette Score for EM: {em_silhouette:.4f}')

# Treatment suggestions based on clusters
treatment_suggestions = {
    0: "Lifestyle changes and regular exercise.",
    1: "Medication management and regular monitoring.",
    2: "Nutritional counseling and weight management."
}

data['Treatment_Suggestion'] = data['KMeans_Cluster'].map(treatment_suggestions)

# Plotting clusters
plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

# EM (Gaussian Mixture) plot
plt.subplot(1, 2, 2)
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=em_labels, cmap='viridis')
plt.title('EM Clustering')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

plt.show()
