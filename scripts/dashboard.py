import streamlit as st
import pandas as pd
import os

# Define the path to the 'customer_agg.csv' file
csv_path = r'C:/Users/user/Desktop/Github/TelecomUserAnalysis/scripts/customer_agg.csv'

# Check if the file exists
if not os.path.isfile(csv_path):
    st.error(f"File not found: {csv_path}")
else:
    # Load the data
    customer_agg = pd.read_csv(csv_path)

    # Title of the Dashboard
    st.title('Telecom User Experience Analysis Dashboard')

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    options = st.sidebar.radio("Select an option:", ["Overview", "Clustering Analysis", "Handset Type Analysis"])

    if options == "Overview":
        st.header("Overview")
        st.write("This dashboard provides insights into telecom user experience based on aggregated data.")
        
        st.subheader("Data Overview")
        st.write(customer_agg.head())
        
        st.subheader("Top 10 TCP Retransmission Values")
        st.write(customer_agg[['Bearer Id', 'Avg_TCP_retransmission']].nlargest(10, 'Avg_TCP_retransmission'))
        
        st.subheader("Top 10 RTT Values")
        st.write(customer_agg[['Bearer Id', 'Avg_RTT']].nlargest(10, 'Avg_RTT'))
        
        st.subheader("Top 10 Throughput Values")
        st.write(customer_agg[['Bearer Id', 'Avg_Throughput']].nlargest(10, 'Avg_Throughput'))

    elif options == "Clustering Analysis":
        st.header("Clustering Analysis")
        
        if 'Cluster' in customer_agg.columns:
            st.subheader("Cluster Distribution")
            cluster_counts = customer_agg['Cluster'].value_counts()
            st.bar_chart(cluster_counts)
            
            st.subheader("Cluster Centers")
            cluster_centers = customer_agg.groupby('Cluster').mean()
            st.write(cluster_centers)
            
            st.subheader("Cluster Summary")
            for cluster in range(3):
                st.write(f"Cluster {cluster} Summary:")
                st.write(customer_agg[customer_agg['Cluster'] == cluster].describe())
        else:
            st.error("Cluster column not found in data. Please ensure the clustering has been performed in the main script.")

    elif options == "Handset Type Analysis":
        st.header("Handset Type Analysis")
        
        st.subheader("Distribution of Average Throughput per Handset Type")
        throughput_distribution = customer_agg.groupby('Mode_Handset_type')['Avg_Throughput'].describe()
        st.write(throughput_distribution)
        
        st.subheader("Average TCP Retransmission per Handset Type")
        tcp_distribution = customer_agg.groupby('Mode_Handset_type')['Avg_TCP_retransmission'].describe()
        st.write(tcp_distribution)

        # Interactive filtering by handset type
        selected_handset = st.selectbox("Select Handset Type", customer_agg['Mode_Handset_type'].unique())
        filtered_data = customer_agg[customer_agg['Mode_Handset_type'] == selected_handset]
        
        st.subheader(f"Details for {selected_handset}")
        st.write(filtered_data)

    # Add a footer
    st.write("### Footer")
    st.write("Developed by [Your Name] - [Your Contact Information]")

# This is a placeholder for deployment instructions. Ensure your environment is configured to deploy Streamlit apps.
