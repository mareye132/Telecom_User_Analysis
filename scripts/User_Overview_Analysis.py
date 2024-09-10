import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import psycopg2

# Database connection and data loading function
def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='Maru@132',
        host='localhost',
        port='5432'
    )
    return conn

def load_data_from_postgres():
    """Load data from PostgreSQL database."""
    conn = get_db_connection()
    query = "SELECT * FROM xdr_data"  # Modify with your actual table name
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Columns in DataFrame:", df.columns.tolist())  # Print column names for debugging
    return df

# 1.1: Aggregation of user behavior on applications
def aggregate_user_behavior(df):
    """Aggregate user behavior information."""
    if 'MSISDN/Number' not in df.columns:
        raise KeyError("Column 'MSISDN/Number' not found in DataFrame")
    aggregated = df.groupby('MSISDN/Number').agg({
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum',
        'Social Media DL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum'
    }).reset_index()
    return aggregated

# 1.2: Top 10 Handsets
def top_10_handsets(df):
    """Identify the top 10 handsets used by customers."""
    if 'Handset Type' not in df.columns:
        raise KeyError("Column 'Handset Type' not found in DataFrame")
    top_handsets = df['Handset Type'].value_counts().head(10)
    return top_handsets

# 1.2: Top 3 Handset Manufacturers
def top_3_manufacturers(df):
    """Identify the top 3 handset manufacturers."""
    if 'Handset Manufacturer' not in df.columns:
        raise KeyError("Column 'Handset Manufacturer' not found in DataFrame")
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    return top_manufacturers

# 1.2: Top 5 Handsets per Manufacturer
def top_5_handsets_per_manufacturer(df):
    """Identify the top 5 handsets per top 3 manufacturers."""
    if 'Handset Manufacturer' not in df.columns or 'Handset Type' not in df.columns:
        raise KeyError("Columns 'Handset Manufacturer' or 'Handset Type' not found in DataFrame")
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index
    top_5_handsets = {}
    for manufacturer in top_manufacturers:
        top_5_handsets[manufacturer] = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
    return top_5_handsets

# 1.2: Handle missing values and outliers
def handle_missing_outliers(df):
    """Handle missing values and outliers."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Outlier handling (e.g., z-score method) can be added here
    return df

# 1.2: Variable transformation (Segment into decile classes)
def segment_users_by_duration(df):
    """Segment users into decile classes based on total session duration."""
    duration_col = 'Activity Duration DL (ms)'
    if duration_col not in df.columns:
        raise KeyError(f"Column '{duration_col}' not found in DataFrame")
    df['duration_decile'] = pd.qcut(df[duration_col], 10, labels=False) + 1
    return df

# 1.2: Univariate Non-Graphical Analysis
def non_graphical_univariate_analysis(df):
    """Perform non-graphical univariate analysis."""
    summary_stats = df.describe()
    print("Summary Statistics:\n", summary_stats)  # Added print statement
    return summary_stats

# 1.2: Graphical Univariate Analysis
def graphical_univariate_analysis(df):
    """Perform graphical univariate analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Activity Duration DL (ms)'], ax=ax)
    plt.title('Distribution of Activity Duration DL')
    plt.show()

# 1.2: Bivariate Analysis
def bivariate_analysis(df):
    """Explore relationships between total data (DL+UL) and different applications."""
    if 'Total DL (Bytes)' not in df.columns or 'Total UL (Bytes)' not in df.columns:
        raise KeyError("Columns 'Total DL (Bytes)' or 'Total UL (Bytes)' not found in DataFrame")
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    total_data = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    for app in applications:
        if app not in df.columns:
            raise KeyError(f"Column '{app}' not found in DataFrame")
        sns.scatterplot(x=total_data, y=df[app])
        plt.title(f'Total Data vs {app.capitalize()} Usage')
        plt.xlabel('Total Data (DL+UL)')
        plt.ylabel(f'{app.capitalize()} Usage')
        plt.show()

# 1.2: Correlation Analysis
def correlation_analysis(df):
    """Compute correlation matrix for the applications."""
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    if not all(app in df.columns for app in applications):
        raise KeyError("One or more columns in applications list not found in DataFrame")
    correlation_matrix = df[applications].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Analysis of Application Usage')
    plt.show()

# 1.2: PCA (Dimensionality Reduction)
def pca_analysis(df):
    """Perform Principal Component Analysis (PCA) and plot the results."""
    applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    if not all(app in df.columns for app in applications):
        raise KeyError("One or more columns in applications list not found in DataFrame")
    pca_features = df[applications]
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(pca_features)
    
    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - Application Usage')
    plt.show()
    print("PCA Explained Variance Ratios:", pca.explained_variance_ratio_)
    return pca

# Main script execution
if __name__ == "__main__":
    # Load data
    df = load_data_from_postgres()

    # Handle missing values and outliers
    df = handle_missing_outliers(df)

    # Aggregate user behavior
    aggregated_df = aggregate_user_behavior(df)
    print("Aggregated User Behavior:\n", aggregated_df.head())

    # Top 10 handsets
    print("Top 10 Handsets:\n", top_10_handsets(df))

    # Top 3 manufacturers and top 5 handsets per manufacturer
    top_manufacturers = top_3_manufacturers(df)
    print("Top 3 Handset Manufacturers:\n", top_manufacturers)
    top_handsets_per_manufacturer = top_5_handsets_per_manufacturer(df)
    print("Top 5 Handsets per Manufacturer:\n", top_handsets_per_manufacturer)

    # Segment users by duration
    df = segment_users_by_duration(df)

    # Univariate Analysis
    non_graphical_univariate_analysis(df)
    graphical_univariate_analysis(df)

    # Bivariate Analysis
    bivariate_analysis(df)

    # Correlation Analysis
    correlation_analysis(df)

    # PCA
    pca_analysis(df)
