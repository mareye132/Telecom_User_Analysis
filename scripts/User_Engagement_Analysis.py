import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import psycopg2
# Database connection function
def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn_string = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password='Maru@132',
            host='localhost',
            port='5432'
        )
        return conn_string
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def load_data_from_postgres():
    """Load data from a PostgreSQL database."""
    conn_string = get_db_connection()
    if conn_string:
        query = "SELECT * FROM xdr_data"
        df = pd.read_sql_query(query, conn_string)
        return df
    else:
        return None

def aggregate_engagement_metrics(df):
    """Aggregate engagement metrics per customer ID."""
    if 'MSISDN/Number' not in df.columns or 'Bearer Id' not in df.columns:
        print("Required columns are missing from the DataFrame.")
        return None, None, None, None

    df['Session Duration'] = df['Activity Duration DL (ms)'] + df['Activity Duration UL (ms)']
    df['Session Frequency'] = df.groupby('MSISDN/Number')['Bearer Id'].transform('count')
    df['Total Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    engagement_metrics = df.groupby('MSISDN/Number').agg({
        'Session Frequency': 'mean',
        'Session Duration': 'mean',
        'Total Traffic': 'sum'
    }).reset_index()
    
    top_10_frequency = engagement_metrics.nlargest(10, 'Session Frequency')
    top_10_duration = engagement_metrics.nlargest(10, 'Session Duration')
    top_10_traffic = engagement_metrics.nlargest(10, 'Total Traffic')

    return engagement_metrics, top_10_frequency, top_10_duration, top_10_traffic

def kmeans_clustering(df):
    """Perform KMeans clustering."""
    if df is None:
        return None, None, None
    
    # Normalize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Session Frequency', 'Session Duration', 'Total Traffic']])
    
    # Determine the optimal value of k using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    # Plot elbow method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # Based on the elbow plot, choose the optimal k
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(df_scaled)
    
    df['Cluster'] = y_kmeans
    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = df['Cluster'].value_counts()

    return df, cluster_centers, cluster_sizes

def compute_cluster_stats(df):
    """Compute statistics for each cluster."""
    cluster_stats = df.groupby('Cluster').agg({
        'Session Frequency': ['min', 'max', 'mean', 'sum'],
        'Session Duration': ['min', 'max', 'mean', 'sum'],
        'Total Traffic': ['min', 'max', 'mean', 'sum']
    })
    return cluster_stats

def plot_top_applications(df):
    """Plot the top 3 most used applications."""
    if 'Google DL (Bytes)' not in df.columns or 'Social Media DL (Bytes)' not in df.columns:
        print("Required columns are missing from the DataFrame.")
        return
    
    app_traffic = df[['Google DL (Bytes)', 'Social Media DL (Bytes)', 'Email DL (Bytes)']].sum()
    top_apps = app_traffic.nlargest(3)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_apps.index, y=top_apps.values)
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')
    plt.show()

# Main code
if __name__ == "__main__":
    df = load_data_from_postgres()

    if df is not None:
        # Inspect columns
        print("Column Names in DataFrame:", df.columns)

        # Aggregate engagement metrics
        engagement_metrics, top_10_frequency, top_10_duration, top_10_traffic = aggregate_engagement_metrics(df)
        
        if engagement_metrics is not None:
            print("Top 10 Customers by Session Frequency:")
            print(top_10_frequency)
            print("\nTop 10 Customers by Session Duration:")
            print(top_10_duration)
            print("\nTop 10 Customers by Total Traffic:")
            print(top_10_traffic)

            # Perform KMeans clustering
            df_clustered, cluster_centers, cluster_sizes = kmeans_clustering(engagement_metrics)
            if df_clustered is not None:
                print("\nCluster Centers:")
                print(cluster_centers)
                print("\nCluster Sizes:")
                print(cluster_sizes)

                # Compute cluster statistics
                cluster_stats = compute_cluster_stats(df_clustered)
                print("\nCluster Statistics:")
                print(cluster_stats)

                # Plot top applications
                plot_top_applications(df)
