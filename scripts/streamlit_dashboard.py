# app.py

import streamlit as st
import pandas as pd
from data_processing import (
    load_data_from_postgres, top_10_handsets, top_3_manufacturers,
    top_5_handsets_per_manufacturer, interpretation_and_recommendation,
    non_graphical_univariate_analysis, graphical_univariate_analysis,
    bivariate_analysis, correlation_analysis, pca_analysis,
    handle_missing_outliers, segment_users_by_duration
)

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

    st.subheader('Interpretation and Recommendation')
    interpretation = interpretation_and_recommendation()
    st.write(interpretation)

    st.subheader('Data Overview')
    st.write(df.head())

    st.subheader('Univariate Analysis')
    st.write(non_graphical_univariate_analysis(df))
    graphical_univariate_analysis(df)

    st.subheader('Bivariate Analysis')
    bivariate_analysis(df)

    st.subheader('Correlation Analysis')
    correlation_analysis(df)

    st.subheader('PCA Analysis')
    pca_analysis(df)

# Main code
if __name__ == "__main__":
    df = load_data_from_postgres()
    df = handle_missing_outliers(df)
    df = segment_users_by_duration(df)
    streamlit_dashboard(df)
