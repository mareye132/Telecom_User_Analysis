# importing neccessary libraries.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    return pd.read_csv('your_dataset.csv')

st.title("Telecom User Overview Dashboard")

# Load the data
data = load_data()

# Visualize top handsets
top_handsets = data['handset'].value_counts().head(10)
st.write("Top 10 Handsets")
st.bar_chart(top_handsets)

# Visualize correlation matrix
st.write("Correlation Matrix")
correlation_matrix = data.corr()
st.write(correlation_matrix)

# More visualizations...
