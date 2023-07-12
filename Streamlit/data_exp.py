import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(url, header = None, names = column_names)
    return data

iris_data = load_iris_data()

st.title("Iris Dataset Explorer")
st.sidebar.title("Explore Options")
st.set_page_config(layout = "wide")

# Display the raw data
st.sidebar.subheader("Raw Data")
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(iris_data)

# Question 1: Show the average sepal length for each species
st.sidebar.subheader("Average Sepal Length")
if st.sidebar.checkbox("Show Average Sepal Length"):
    avg_sepal_length = iris_data.groupby("species")["sepal_length'="].mean()
    st.write(avg_sepal_length)

# Question 2: Display a scatter plot comparing two features
st.sidebar.subheader("Scatter Plot")
if st.sidebar.checkbox("Show Scatter Plot"):
    feature1 = st.sidebar.selectbox("Select Feature 1", iris_data.columns[:-1])
    feature2 = st.sidebar.selectbox("Select Feature 2", iris_data.columns[:-1])
    st.write("Scatter Plot:", feature1, "vs", feature2)
    plt.figure(figsize = (8, 6))
    sns.scatterplot(data = iris_data, x = feature1, y = feature2, hue = "species")
    st.pyplot()

# Question 3: Filter data based on species
st.sidebar.subheader("Filter Data")
selected_species = st.sidebar.multiselect("Select Species", iris_data["species"].unique())
filtered_data = iris_data[iris_data["species"].isin(selected_species)]
if len(filtered_data) > 0:
    st.write("Filtered Data:")
    st.write(filtered_data)
else:
    st.warning("No data available for the selected species.")

# Question 4: Display a pairplot for the selected species
st.sidebar.subheader("Pairplot")
if st.sidebar.checkbox("Show Pairplot"):
    st.write("Pairplot for Selected Species")
    sns.pairplot(filtered_data, hue = "species")
    st.pyplot()

# Question 5: Show the distribution of a selected feature
st.sidebar.subheader("Feature Distribution")
feature = st.sidebar.selectbox("Select Feature", iris_data.columns[:-1])
st.write("Distribution of", feature)
sns.histplot(data = iris_data, x = feature, hue = "species", element = "step", kde = True)
st.pyplot()
