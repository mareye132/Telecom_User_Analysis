import os
import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Handle missing values and outliers
df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)

# Remove outliers based on TCP DL Retrans. Vol (Bytes)
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

# Define the path to the 'scripts' directory
scripts_dir = r'C:/Users/user/Desktop/Github/TelecomUserAnalysis/scripts'

# Create the directory if it doesn't exist
if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)

# Save the aggregated DataFrame as CSV in the 'scripts' directory
csv_path = os.path.join(scripts_dir, 'customer_agg.csv')
customer_agg.to_csv(csv_path, index=False)

print(f"Aggregated customer data saved to {csv_path}")

# Task 3.2 - Compute & list top, bottom, and most frequent values
top_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
bottom_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
most_frequent_tcp = df['TCP DL Retrans. Vol (Bytes)'].mode().head(10)

top_10_rtt = df['Avg RTT DL (ms)'].nlargest(10)
bottom_10_rtt = df['Avg RTT DL (ms)'].nsmallest(10)
most_frequent_rtt = df['Avg RTT DL (ms)'].mode().head(10)

top_10_throughput = df['Avg Bearer TP DL (kbps)'].nlargest(10)
bottom_10_throughput = df['Avg Bearer TP DL (kbps)'].nsmallest(10)
most_frequent_throughput = df['Avg Bearer TP DL (kbps)'].mode().head(10)

print("Top 10 TCP Retransmission Values:\n", top_10_tcp)
print("Bottom 10 TCP Retransmission Values:\n", bottom_10_tcp)
print("Most Frequent TCP Retransmission Values:\n", most_frequent_tcp)

print("Top 10 RTT Values:\n", top_10_rtt)
print("Bottom 10 RTT Values:\n", bottom_10_rtt)
print("Most Frequent RTT Values:\n", most_frequent_rtt)

print("Top 10 Throughput Values:\n", top_10_throughput)
print("Bottom 10 Throughput Values:\n", bottom_10_throughput)
print("Most Frequent Throughput Values:\n", most_frequent_throughput)

# Task 3.3 - Compute & report distributions
# Distribution of Average Throughput per Handset Type
distribution_throughput = customer_agg.groupby('Mode_Handset_type')['Avg_Throughput'].describe()
print("Distribution of Average Throughput per Handset Type:\n", distribution_throughput)

# Average TCP Retransmission per Handset Type
distribution_tcp = customer_agg.groupby('Mode_Handset_type')['Avg_TCP_retransmission'].describe()
print("Average TCP Retransmission per Handset Type:\n", distribution_tcp)

# Task 3.4 - K-Means Clustering
features = customer_agg[['Avg_TCP_retransmission', 'Avg_RTT', 'Avg_Throughput']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=0)
customer_agg['Cluster'] = kmeans.fit_predict(features_scaled)

# Save the updated DataFrame with clusters to CSV
customer_agg.to_csv(csv_path, index=False)

print(f"Updated customer data with clusters saved to {csv_path}")

# Brief description of each cluster
for cluster in range(3):
    print(f"Cluster {cluster} Summary:")
    print(customer_agg[customer_agg['Cluster'] == cluster].describe())
