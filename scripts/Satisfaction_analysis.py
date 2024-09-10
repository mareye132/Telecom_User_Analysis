# Import necessary libraries
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LinearRegression
import numpy as np
import mlflow  # For model tracking (MlOps)
import os

# Database connection parameters for PostgreSQL
db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'Maru@132',
    'host': 'localhost',
    'port': '5432'
}

# Connect to PostgreSQL
conn_postgres = psycopg2.connect(
    dbname=db_params['dbname'],
    user=db_params['user'],
    password=db_params['password'],
    host=db_params['host'],
    port=db_params['port']
)
cursor = conn_postgres.cursor()

# Check existing columns in xdr_data table
cursor.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'xdr_data'
""")
columns = cursor.fetchall()
columns = [col[0] for col in columns]
print("Current columns in xdr_data table:", columns)

# Ensure required columns exist
required_columns = ['engagement_score', 'experience_score', 'satisfaction_score', 'eng_exp_cluster']
missing_columns = [col for col in required_columns if col not in columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
    for col in missing_columns:
        if col == 'engagement_score':
            cursor.execute("ALTER TABLE xdr_data ADD COLUMN engagement_score FLOAT;")
        elif col == 'experience_score':
            cursor.execute("ALTER TABLE xdr_data ADD COLUMN experience_score FLOAT;")
        elif col == 'satisfaction_score':
            cursor.execute("ALTER TABLE xdr_data ADD COLUMN satisfaction_score FLOAT;")
        elif col == 'eng_exp_cluster':
            cursor.execute("ALTER TABLE xdr_data ADD COLUMN eng_exp_cluster INT;")
    conn_postgres.commit()

# Query to fetch data from PostgreSQL
query = "SELECT * FROM xdr_data"
df = pd.read_sql_query(query, conn_postgres)

# Handle missing values
df['TCP DL Retrans. Vol (Bytes)'] = df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean())
df['Avg RTT DL (ms)'] = df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean())
df['Avg Bearer TP DL (kbps)'] = df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean())
df['Handset Type'] = df['Handset Type'].fillna(df['Handset Type'].mode()[0])

# Feature engineering
features = df[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]

# Define example centroid values 
engagement_centroid = np.array([[50000, 20, 1000]])  
experience_centroid = np.array([[60000, 25, 1500]])  

# Task 4.1 - Calculate engagement and experience scores
df['Engagement_Score'] = euclidean_distances(features, engagement_centroid).flatten()
df['Experience_Score'] = euclidean_distances(features, experience_centroid).flatten()

# Task 4.2 - Calculate satisfaction score
df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

# Report top 10 satisfied customers
top_10_satisfied = df[['Bearer Id', 'Satisfaction_Score']].sort_values(by='Satisfaction_Score', ascending=False).head(10)
print("Top 10 Satisfied Customers:")
print(top_10_satisfied)

# Task 4.3 - Regression model to predict satisfaction score
X = df[['Engagement_Score', 'Experience_Score']]
y = df['Satisfaction_Score']
model = LinearRegression()

# Start model tracking with MLflow
mlflow.start_run()

model.fit(X, y)
print("Regression Coefficients:")
print(model.coef_)

# Log the model and parameters with MLflow
mlflow.log_param("regression_model", "LinearRegression")
mlflow.log_param("input_features", X.columns.tolist())
mlflow.log_metric("r_squared", model.score(X, y))

# End model tracking
mlflow.end_run()

# Task 4.4 - K-means clustering on engagement and experience scores
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
df['Eng_Exp_Cluster'] = kmeans.labels_

# Task 4.5 - Aggregate average satisfaction & experience score per cluster
cluster_summary = df.groupby('Eng_Exp_Cluster').agg({
    'Satisfaction_Score': 'mean',
    'Experience_Score': 'mean'
}).reset_index()
print("Cluster Summary:")
print(cluster_summary)

# Task 4.6 - Export the table to PostgreSQL (instead of MySQL)
# Insert the results back into PostgreSQL
insert_query_pg = """
INSERT INTO user_scores (Bearer_Id, Engagement_Score, Experience_Score, Satisfaction_Score, Eng_Exp_Cluster)
VALUES (%s, %s, %s, %s, %s)
"""

# Create table if it doesn't exist in PostgreSQL
create_table_pg = """
CREATE TABLE IF NOT EXISTS user_scores (
    Bearer_Id VARCHAR(50),
    Engagement_Score FLOAT,
    Experience_Score FLOAT,
    Satisfaction_Score FLOAT,
    Eng_Exp_Cluster INT
)
"""
cursor.execute(create_table_pg)
conn_postgres.commit()

# Insert data into PostgreSQL
for _, row in df.iterrows():
    cursor.execute(insert_query_pg, (
        str(row['Bearer Id']), 
        float(row['Engagement_Score']),
        float(row['Experience_Score']),
        float(row['Satisfaction_Score']),
        int(row['Eng_Exp_Cluster'])
    ))
conn_postgres.commit()

# Verify data in PostgreSQL (run a select query)
cursor.execute("SELECT * FROM user_scores LIMIT 5")
result = cursor.fetchall()
print("Data in PostgreSQL (first 5 rows):", result)

# Clean up
cursor.close()
conn_postgres.close()

# Task 4.7 - Model deployment tracking using Docker/MlOps tool (mlflow setup is done above)
print("To deploy and track models, use Docker or any CI/CD tools to create an image and track changes.")

# Script completion
print("Satisfaction Analysis script completed successfully.")
