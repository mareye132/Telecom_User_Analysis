import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import streamlit as st

# Function to load data from a CSV file
def load_data_from_csv():
    """Load data from a CSV file."""
    path = 'C:/Users/user/Desktop/Github/TelecomUserAnalysis/data/challenge_data_source.csv'
    df = pd.read_csv(path)
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

# 1.2: Interpretation and Recommendation
def interpretation_and_recommendation():
    """Make a short interpretation and recommendation for the marketing team."""
    interpretation = """
    The top 10 handsets are popular models that should be targeted in marketing campaigns, as they indicate user preferences.
    The top 3 manufacturers dominate the market, and understanding their top-performing models could help in negotiating marketing deals.
    The top 5 handsets per manufacturer indicate popular trends in handset usage, which the marketing team can leverage for promoting data plans.
    """
    return interpretation

# 1.2: Handle missing values and outliers
def handle_missing_outliers(df):
    """Handle missing values and outliers."""
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Fill missing values in numeric columns with the mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Optionally handle outliers here
    
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
    return pca

# Streamlit Dashboard Integration
def streamlit_dashboard(df):
    """Interactive Streamlit dashboard to visualize the analysis."""
    st.title('Telecom User Overview & Analysis')

    st.subheader('Top 10 Handsets')
    top_handsets = top_10_handsets(df)
    st.write(top_handsets)

    st.subheader('Top 3 Handset Manufacturers')
    top_manufacturers = top_3_manufacturers(df)
    st.write(top_manufacturers)

    st.subheader('Top 5 Handsets per Manufacturer')
    top_5_handsets = top_5_handsets_per_manufacturer(df)
    for manufacturer, handsets in top_5_handsets.items():
        st.write(f'{manufacturer}:')
        st.write(handsets)

    st.subheader('Interpretation & Recommendation')
    st.write(interpretation_and_recommendation())

    st.subheader('Univariate Analysis')
    univariate_summary = non_graphical_univariate_analysis(df)
    st.write(univariate_summary)

    st.subheader('Graphical Univariate Analysis')
    graphical_univariate_analysis(df)

    st.subheader('Bivariate Analysis')
    bivariate_analysis(df)

    st.subheader('Correlation Analysis')
    correlation_analysis(df)

    st.subheader('PCA Analysis')
    pca = pca_analysis(df)

# Main function to run the analysis
def main():
    df = load_data_from_csv()
    df = handle_missing_outliers(df)
    df = segment_users_by_duration(df)
    streamlit_dashboard(df)

if __name__ == "__main__":
    main()