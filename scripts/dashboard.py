import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
customer_agg = pd.read_csv('customer_agg.csv')
top_tcp = pd.read_csv('top_tcp.csv')
bottom_tcp = pd.read_csv('bottom_tcp.csv')
freq_tcp = pd.read_csv('freq_tcp.csv')

top_rtt = pd.read_csv('top_rtt.csv')
bottom_rtt = pd.read_csv('bottom_rtt.csv')
freq_rtt = pd.read_csv('freq_rtt.csv')

top_throughput = pd.read_csv('top_throughput.csv')
bottom_throughput = pd.read_csv('bottom_throughput.csv')
freq_throughput = pd.read_csv('freq_throughput.csv')

customer_clusters = pd.read_csv('customer_clusters.csv')

st.title('Telecom User Experience Analytics')

# Task 3.1 - Aggregates
st.header('Aggregates per Customer')
st.write(customer_agg)

# Task 3.2 - Top, Bottom, Frequent Values
st.header('Top 10 TCP Retransmission Values')
st.write(top_tcp)

st.header('Bottom 10 TCP Retransmission Values')
st.write(bottom_tcp)

st.header('Most Frequent TCP Retransmission Values')
st.write(freq_tcp)

# Distribution Plots for Task 3.3
st.header('Distribution of Average Throughput per Handset Type')
fig, ax = plt.subplots()
sns.boxplot(x='Handset_type', y='Avg_Throughput', data=customer_agg, ax=ax)
st.pyplot(fig)

st.header('Average TCP Retransmission per Handset Type')
fig, ax = plt.subplots()
sns.boxplot(x='Handset_type', y='Avg_TCP_retransmission', data=customer_agg, ax=ax)
st.pyplot(fig)

# K-Means Clustering Results
st.header('K-Means Clustering Results')
st.write(customer_clusters.groupby('Cluster').mean())

# Add additional interactive elements and visualizations as needed
