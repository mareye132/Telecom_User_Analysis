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
# st.write("Loaded dataset columns:", customer_agg.columns.tolist())

# Function to display crossbonding results
def display_crossbonding_results(df):
    st.subheader("Crossbonding Data Analysis")  # Adding a clear label
    if 'Cluster' not in df.columns:
        #st.warning("The 'Cluster' column is missing in the dataset. Displaying available data instead.")
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
    st.subheader("Additional Data Visualizations")  # Adding a clear label
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

# Interactive Filters and Widgets
def filter_data(df):
    st.sidebar.subheader("Data Filters")

    # Adding filters for numeric columns if available
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_columns:
        selected_column = st.sidebar.selectbox("Select a Numeric Column for Filtering", numeric_columns)
        min_value = df[selected_column].min()
        max_value = df[selected_column].max()
        selected_range = st.sidebar.slider(
            f"Select Range for {selected_column}", min_value, max_value, (min_value, max_value)
        )
        # Filtering the data
        filtered_df = df[(df[selected_column] >= selected_range[0]) & (df[selected_column] <= selected_range[1])]
        return filtered_df
    else:
        st.sidebar.warning("No numeric columns available for filtering.")
        return df

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option:", ["Crossbonding Results", "Additional Visualizations", "Data Filtering"])

# Main Title
st.title("Customer Aggregation Dashboard")

# Display content based on selected option
if option == "Crossbonding Results":
    st.info("View crossbonding data insights")
    display_crossbonding_results(customer_agg)
elif option == "Additional Visualizations":
    st.info("View additional data visualizations")
    display_additional_visualizations(customer_agg)
elif option == "Data Filtering":
    st.info("Filter data based on specific criteria")
    filtered_data = filter_data(customer_agg)
    st.write(filtered_data)
