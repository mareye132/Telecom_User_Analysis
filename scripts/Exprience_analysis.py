# importing neccessary liberaries
import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# PostgreSQL connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'Maru@132',
    'host': 'localhost',
    'port': '5432'
}

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=db_params['dbname'],
    user=db_params['user'],
    password=db_params['password'],
    host=db_params['host'],
    port=db_params['port']
)

# Load data from PostgreSQL
query = "SELECT * FROM xdr_data"
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# Print column names to check and update names in the script
print("Columns in DataFrame:", df.columns)

# Handle missing values and outliers
df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)

# Remove outliers (Example for TCP retransmission)
df = df[(np.abs(df['TCP DL Retrans. Vol (Bytes)'] - df['TCP DL Retrans. Vol (Bytes)'].mean()) <= (3 * df['TCP DL Retrans. Vol (Bytes)'].std()))]

# Task 3.1 - Aggregate information
customer_agg = df.groupby('Bearer Id').agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'Avg RTT DL (ms)': 'mean',
    'Handset Type': lambda x: x.mode()[0],
    'Avg Bearer TP DL (kbps)': 'mean'
}).reset_index()

customer_agg.rename(columns={
    'TCP DL Retrans. Vol (Bytes)': 'Avg_TCP_retransmission',
    'Avg RTT DL (ms)': 'Avg_RTT',
    'Handset Type': 'Mode_Handset_type',
    'Avg Bearer TP DL (kbps)': 'Avg_Throughput'
}, inplace=True)

# Task 3.2 - Compute & list top, bottom, and most frequent values
top_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
bottom_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
freq_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].mode().head(10)

top_10_rtt = df['Avg RTT DL (ms)'].nlargest(10)
bottom_10_rtt = df['Avg RTT DL (ms)'].nsmallest(10)
freq_10_rtt = df['Avg RTT DL (ms)'].mode().head(10)

top_10_throughput = df['Avg Bearer TP DL (kbps)'].nlargest(10)
bottom_10_throughput = df['Avg Bearer TP DL (kbps)'].nsmallest(10)
freq_10_throughput = df['Avg Bearer TP DL (kbps)'].mode().head(10)

print("Top 10 TCP retransmission values:\n", top_10_tcp)
print("Bottom 10 TCP retransmission values:\n", bottom_10_tcp)
print("Most frequent TCP retransmission values:\n", freq_10_tcp)

print("Top 10 RTT values:\n", top_10_rtt)
print("Bottom 10 RTT values:\n", bottom_10_rtt)
print("Most frequent RTT values:\n", freq_10_rtt)

print("Top 10 Throughput values:\n", top_10_throughput)
print("Bottom 10 Throughput values:\n", bottom_10_throughput)
print("Most frequent Throughput values:\n", freq_10_throughput)

# Task 3.3 - Compute and report distribution of average throughput per handset type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Mode_Handset_type', y='Avg_Throughput', data=customer_agg)
plt.title('Distribution of Average Throughput per Handset Type')
plt.xticks(rotation=45)
plt.show()

# Task 3.3 - Average TCP retransmission per handset type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Mode_Handset_type', y='Avg_TCP_retransmission', data=customer_agg)
plt.title('Average TCP Retransmission per Handset Type')
plt.xticks(rotation=45)
plt.show()

# Task 3.4 - K-Means Clustering
features = customer_agg[['Avg_TCP_retransmission', 'Avg_RTT', 'Avg_Throughput']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=0)
customer_agg['Cluster'] = kmeans.fit_predict(features_scaled)

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels:\n", customer_agg['Cluster'].value_counts())

# Brief description of each cluster
for cluster in range(3):
    print(f"Cluster {cluster} Summary:")
    print(customer_agg[customer_agg['Cluster'] == cluster].describe())
