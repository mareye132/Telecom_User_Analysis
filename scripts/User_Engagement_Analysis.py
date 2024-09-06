import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_from_csv(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def basic_metrics(df):
    """Compute basic metrics for each numeric column."""
    numeric_df = df.select_dtypes(include='number')
    return {
        'Mean': numeric_df.mean(),
        'Median': numeric_df.median(),
        'Standard Deviation': numeric_df.std(),
        'Variance': numeric_df.var()
    }

def non_graphical_univariate_analysis(df, quantitative_vars):
    """Compute dispersion parameters for each quantitative variable."""
    print("DataFrame columns:", df.columns)
    
    # Filter quantitative_vars based on actual column names
    quantitative_vars = [var for var in quantitative_vars if var in df.columns]
    if not quantitative_vars:
        raise ValueError("None of the specified quantitative variables were found in the DataFrame.")
    
    quantitative_df = df[quantitative_vars].select_dtypes(include='number')
    dispersion_params = {
        'Variance': quantitative_df.var(),
        'Skewness': quantitative_df.skew(),
        'Kurtosis': quantitative_df.kurt()
    }
    return dispersion_params

def graphical_univariate_analysis(df, quantitative_vars):
    """Generate graphical univariate analysis."""
    quantitative_df = df[quantitative_vars].select_dtypes(include='number')
    for column in quantitative_df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(quantitative_df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def bivariate_analysis(df, applications, total_dl_ul):
    """Perform bivariate analysis."""
    if 'Total DL (Bytes)' in df.columns and 'Total UL (Bytes)' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=df['Total DL (Bytes)'], y=df['Total UL (Bytes)'])
        plt.title('Total Download vs Total Upload')
        plt.xlabel('Total Download (Bytes)')
        plt.ylabel('Total Upload (Bytes)')
        plt.show()

def correlation_analysis(df, applications):
    """Perform correlation analysis."""
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix

def dimensionality_reduction(df, applications):
    """Perform dimensionality reduction (placeholder)."""
    print("Dimensionality reduction is not implemented.")
    return []

def aggregate_user_traffic(df):
    """Aggregate user traffic data (placeholder)."""
    print("User traffic aggregation is not implemented.")
    return df

def kmeans_clustering(df):
    """Perform KMeans clustering (placeholder)."""
    print("KMeans clustering is not implemented.")
    return df, None, None

# Example usage
if __name__ == "__main__":
    path = 'C:/Users/user/Desktop/Github/TelecomUserAnalysis/data/challenge_data_source.csv'
    df = load_data_from_csv(path)

    # Update quantitative_vars with actual columns from DataFrame
    quantitative_vars = [
        'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'DL TP < 50 Kbps (%)',
        '50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
        'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)',
        'UL TP > 300 Kbps (%)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
        'Activity Duration UL (ms)', 'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)',
        'Total UL (Bytes)', 'Total DL (Bytes)'
    ]

    print("Basic Metrics:")
    metrics = basic_metrics(df)
    for key, value in metrics.items():
        print(f"{key}:\n{value}\n")

    print("Dispersion Parameters:")
    dispersion_params = non_graphical_univariate_analysis(df, quantitative_vars)
    for key, value in dispersion_params.items():
        print(f"{key}:\n{value}\n")

    graphical_univariate_analysis(df, quantitative_vars)

    # Placeholder for actual applications and total_dl_ul computation
    applications = []  # Define or load actual applications data
    total_dl_ul = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    bivariate_analysis(df, applications, total_dl_ul)

    correlation_matrix = correlation_analysis(df, applications)
    print("Correlation Matrix:")
    print(correlation_matrix)

    explained_variance_ratio = dimensionality_reduction(df, applications)
    print("Explained Variance Ratio:")
    print(explained_variance_ratio)

    user_traffic = aggregate_user_traffic(df)
    user_traffic, cluster_centers, cluster_sizes = kmeans_clustering(user_traffic)
    print("Cluster Centers:")
    print(cluster_centers)
    print("Cluster Sizes:")
    print(cluster_sizes)
