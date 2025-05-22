import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Synthetic dataset creation
np.random.seed(42)
n = 200  # number of applications

# Simulate applicant IDs (with duplicates)
applicant_ids = np.random.choice([f'ID_{i}' for i in range(150)], size=n, replace=True)
base_time = datetime(2025, 5, 1, 8, 0, 0)
timestamps = [base_time + timedelta(minutes=np.random.poisson(10)) for _ in range(n)]
timestamps.sort()
ages = np.random.choice([22, 23, 24, 25, 26, None], size=n, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
departments = np.random.choice(['Engineering', 'Marketing', 'HR', 'Finance'], size=n)
task_completion = np.random.beta(2, 5, size=n)

# Create DataFrame
applications = pd.DataFrame({
    'applicant_id': applicant_ids,
    'submission_time': timestamps,
    'age': ages,
    'department': departments,
    'task_completion_rate': task_completion
})

#Pattern based anomaly flags

# Flag duplicate entries by applicant_id
applications['is_duplicate'] = applications.duplicated(subset=['applicant_id'], keep=False)

# Calculate time difference between submissions to identify rapid submissions
applications = applications.sort_values('submission_time')
applications['time_diff_minutes'] = applications['submission_time'].diff().dt.total_seconds() / 60
applications['rapid_submission'] = applications['time_diff_minutes'] < 1
applications['inconsistent_data'] = applications['age'].isnull() | (applications['age'] < 18) | (applications['age'] > 60)

#ML anomaly detection 

# Encode department as numeric
applications['department_enc'] = applications['department'].astype('category').cat.codes

# Select features and fill missing ages with median
features = applications[['age', 'task_completion_rate', 'department_enc']].copy()
features['age'].fillna(features['age'].median(), inplace=True)

#Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
applications['iso_forest_anomaly'] = iso_forest.fit_predict(features)

#K-Means clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
applications['cluster'] = kmeans.fit_predict(features)

# Calculate distance from cluster centers
centers = kmeans.cluster_centers_

def dist_from_center(row):
    center = centers[row['cluster']]
    return np.linalg.norm(row[['age', 'task_completion_rate', 'department_enc']] - center)

applications['dist_from_center'] = applications.apply(dist_from_center, axis=1)

# Flag anomalies
threshold = applications['dist_from_center'].quantile(0.95)
applications['kmeans_anomaly'] = applications['dist_from_center'] > threshold

#Pipline to flag suspicious applications
applications['suspicious'] = (
    applications['is_duplicate'] |
    applications['rapid_submission'] |
    applications['inconsistent_data'] |
    (applications['iso_forest_anomaly'] == -1) |
    applications['kmeans_anomaly']
)

suspicious_entries = applications[applications['suspicious']]

print("Sample suspicious applications and anomaly flags:")
print(suspicious_entries[['applicant_id', 'submission_time', 'is_duplicate', 'rapid_submission',
                          'inconsistent_data', 'iso_forest_anomaly', 'kmeans_anomaly']].head(10))
