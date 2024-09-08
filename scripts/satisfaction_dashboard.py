import streamlit as st
import pandas as pd
import psycopg2
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
query = "SELECT * FROM user_scores"
df = pd.read_sql(query, conn)
conn.close()

# Streamlit dashboard
st.title('Customer Satisfaction Analysis Dashboard')

# Display top 10 satisfied customers
st.subheader('Top 10 Satisfied Customers')
top_10_satisfied = df.nlargest(10, 'satisfaction_score')
st.write(top_10_satisfied[['bearer_id', 'satisfaction_score']])

# Plot Satisfaction Scores Distribution
st.subheader('Satisfaction Scores Distribution')
fig, ax = plt.subplots()
sns.histplot(df['satisfaction_score'], kde=True, ax=ax)
st.pyplot(fig)

# Plot Engagement and Experience Scores by Cluster
st.subheader('Engagement and Experience Scores by Cluster')
cluster_summary = df.groupby('eng_exp_cluster').agg({
    'satisfaction_score': 'mean',
    'experience_score': 'mean'
}).reset_index()

fig, ax = plt.subplots()
sns.barplot(x='eng_exp_cluster', y='satisfaction_score', data=cluster_summary, ax=ax, color='blue', label='Satisfaction Score')
sns.barplot(x='eng_exp_cluster', y='experience_score', data=cluster_summary, ax=ax, color='red', label='Experience Score')
ax.legend()
st.pyplot(fig)
