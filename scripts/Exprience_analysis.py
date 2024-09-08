import pandas as pd
import numpy as np
#from sqlalchemy import create_engine
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# PostgreSQL connection parameters


conn_string = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='Maru@132',
        host='localhost',
        port='5432')
# Create a database engine
#engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

# Load data from PostgreSQL
query = "SELECT * FROM xdr_data"
df = pd.read_sql(query, conn_string)

# Handle missing values and outliers
df['TCP_retransmission'].fillna(df['TCP_retransmission'].mean(), inplace=True)
df['RTT'].fillna(df['RTT'].mean(), inplace=True)
df['Throughput'].fillna(df['Throughput'].mean(), inplace=True)
df['Handset_type'].fillna(df['Handset_type'].mode()[0], inplace=True)

# Remove outliers (Example for TCP retransmission)
df = df[(np.abs(df['TCP_retransmission'] - df['TCP_retransmission'].mean()) <= (3 * df['TCP_retransmission'].std()))]

# Task 3.1 - Aggregate information
customer_agg = df.groupby('Customer_ID').agg({
    'TCP_retransmission': 'mean',
    'RTT': 'mean',
    'Handset_type': 'mode',
    'Throughput': 'mean'
}).reset_index()

customer_agg.rename(columns={
    'TCP_retransmission': 'Avg_TCP_retransmission',
    'RTT': 'Avg_RTT',
    'Handset_type': 'Mode_Handset_type',
    'Throughput': 'Avg_Throughput'
}, inplace=True)

# Save aggregated data for dashboard
customer_agg.to_csv('customer_agg.csv', index=False)

# Task 3.2 - Compute top, bottom, and frequent values
top_tcp = df['TCP_retransmission'].nlargest(10)
bottom_tcp = df['TCP_retransmission'].nsmallest(10)
freq_tcp = df['TCP_retransmission'].mode().head(10)

top_rtt = df['RTT'].nlargest(10)
bottom_rtt = df['RTT'].nsmallest(10)
freq_rtt = df['RTT'].mode().head(10)

top_throughput = df['Throughput'].nlargest(10)
bottom_throughput = df['Throughput'].nsmallest(10)
freq_throughput = df['Throughput'].mode().head(10)

# Save these lists to CSV for easy access in the dashboard
top_tcp.to_csv('top_tcp.csv', index=False)
bottom_tcp.to_csv('bottom_tcp.csv', index=False)
freq_tcp.to_csv('freq_tcp.csv', index=False)

top_rtt.to_csv('top_rtt.csv', index=False)
bottom_rtt.to_csv('bottom_rtt.csv', index=False)
freq_rtt.to_csv('freq_rtt.csv', index=False)

top_throughput.to_csv('top_throughput.csv', index=False)
bottom_throughput.to_csv('bottom_throughput.csv', index=False)
freq_throughput.to_csv('freq_throughput.csv', index=False)

# Task 3.3 - Report distributions
# Note: Distribution plots will be done in the dashboard script.

# Task 3.4 - K-Means Clustering
features = customer_agg[['Avg_TCP_retransmission', 'Avg_RTT', 'Avg_Throughput']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
customer_agg['Cluster'] = kmeans.fit_predict(scaled_features)

# Save clustering results
customer_agg.to_csv('customer_clusters.csv', index=False)

print("Tasks completed successfully.")
