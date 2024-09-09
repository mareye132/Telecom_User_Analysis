import streamlit as st
import pandas as pd

# Load the dataset
csv_path = 'scripts/customer_agg.csv'
try:
    customer_agg = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"File not found: {csv_path}")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"The file is empty: {csv_path}")
    st.stop()
except pd.errors.ParserError:
    st.error(f"Error parsing the file: {csv_path}")
    st.stop()

# Display dataset columns for debugging
#st.write("Loaded dataset columns:", customer_agg.columns.tolist())

# Function to display crossbonding results
def display_crossbonding_results(df):
    #st.write("Columns in the dataset:", df.columns.tolist())
    if 'Cluster' not in df.columns:
        #st.error("The 'Cluster' column is missing in the dataset.")
        st.write("Displaying available data:")
        st.write(df)  # Display the entire dataset
        return

    # Check if there are numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_cols.empty:
        st.error("No numeric columns available for aggregation.")
        return

    try:
        # Group by 'Cluster' and calculate the mean of numeric columns
        cluster_centers = df.groupby('Cluster')[numeric_cols].mean()
        st.write("Cluster Centers:")
        st.write(cluster_centers)
    except Exception as e:
        st.error(f"An error occurred while calculating cluster centers: {e}")

# Function to display additional visualizations
def display_additional_visualizations(df):
    if 'Mode_Handset_type' in df.columns and 'Avg_Throughput' in df.columns:
        throughput_by_handset = df.groupby('Mode_Handset_type')['Avg_Throughput'].mean()
        st.write("Distribution of Average Throughput per Handset Type:")
        st.bar_chart(throughput_by_handset)
    else:
        st.warning("Required columns for throughput analysis are missing.")

    if 'Mode_Handset_type' in df.columns and 'Avg_TCP_retransmission' in df.columns:
        retransmission_by_handset = df.groupby('Mode_Handset_type')['Avg_TCP_retransmission'].mean()
        st.write("Average TCP Retransmission per Handset Type:")
        st.bar_chart(retransmission_by_handset)
    else:
        st.warning("Required columns for retransmission analysis are missing.")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option:", ["Crossbonding Results", "Additional Visualizations"])

# Display content based on selected option
if option == "Crossbonding Results":
    st.title("Crossbonding Results")
    display_crossbonding_results(customer_agg)
elif option == "Additional Visualizations":
    st.title("Additional Visualizations")
    display_additional_visualizations(customer_agg)
