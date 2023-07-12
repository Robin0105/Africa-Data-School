import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(url, header = None, names = column_names)
    return data

iris_data = load_iris_data()

# Set app title and page layout
st.title("Iris Dataset Explorer")

# Display the raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(iris_data)

# Show the average sepal length for each species
if st.checkbox("Show Average Sepal Length"):
    avg_sepal_length = iris_data.groupby("species")["sepal_length"].mean()
    st.write(avg_sepal_length)

# Display a scatter plot comparing two features
st.subheader("Compare two features using a scatter plot")
feature_1 = st.selectbox("Select the first feature:", data.columns[:-1])
feature_2 = st.selectbox("Select the second feature:", data.columns[:-1])
scatter_plot = px.scatter(data, x = feature_1, y = feature_2, color = "species", hover_name = "species")
st.plotly_chart(scatter_plot)

# Filter data based on species
st.subheader("Filter Data")
selected_species = st.multiselect("Select Species", iris_data["species"].unique())
filtered_data = iris_data[iris_data["species"].isin(selected_species)]
if len(filtered_data) > 0:
    st.write("Filtered Data:")
    st.write(filtered_data)
else:
    st.warning("No data available for the selected species.")

# Display a pairplot for the selected species
st.subheader("Pairplot")
if st.sidebar.checkbox("Show Pairplot"):
    st.write("Pairplot for Selected Species")
    sns.pairplot(filtered_data, hue = "species")
    st.pyplot()

# Show the distribution of a selected feature
st.subheader("Feature Distribution")
feature = st.selectbox("Select Feature", iris_data.columns[:-1])
st.write("Distribution of", feature)
sns.histplot(data = iris_data, x = feature, hue = "species", element = "step", kde = True)
st.pyplot()
