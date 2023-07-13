import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Titanic Data Analysis")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    data.drop('Cabin', axis = 1, inplace=True)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    return data

data = load_data()

# Data Overview
st.subheader("Data Overview")
st.write(data.head())

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)
  
# Data Summary
st.subheader("Data Summary")
st.write(data.describe())

# Survived vs. Not Survived by Gender
st.subheader("Survived vs. Not Survived by Gender")
survived_gender = data.groupby(['Survived', 'Sex']).size().reset_index(name = 'Count')
fig_gender = px.bar(survived_gender, x = 'Survived', y = 'Count', color = 'Sex', barmode = 'group')
st.plotly_chart(fig_gender)

# Survived vs. Not Survived by Class
st.subheader("Survived vs. Not Survived by Class")
survived_class = data.groupby(['Survived', 'Pclass']).size().reset_index(name = 'Count')
fig_class = px.bar(survived_class, x = 'Survived', y = 'Count', color = 'Pclass', barmode = 'group')
st.plotly_chart(fig_class)

# Age Distribution
st.subheader("Age Distribution of Passengers")
fig_age = px.histogram(data, x = 'Age', nbins = 30)
st.plotly_chart(fig_age)

# Age Distribution by Survival Status
st.subheader("Age Distribution of Passengers by Survival Status")
fig = px.histogram(data, x = 'Age', color = 'Survived', nbins = 30)
st.plotly_chart(fig)

# Fare vs. Class
st.subheader("Fare vs. Class")
fig_fare = px.scatter(data, x = 'Fare', y = 'Pclass', color = 'Survived')
st.plotly_chart(fig_fare)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
correlation = data.corr()
fig_corr = px.imshow(correlation)
st.plotly_chart(fig_corr)
