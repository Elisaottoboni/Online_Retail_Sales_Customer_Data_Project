# python3 -m streamlit run online_retail_streamlit.py

#####################################################
# Libraries
#####################################################

import pandas as pd
import numpy as np
import clean_dataset as cl # I import the Python file that cleans the dataframe
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
import squarify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score


#####################################################
# Reads datasets
#####################################################

df_sales = pd.read_csv('dataset/online_retail.csv')

#####################################################
# Functions
#####################################################

# Function to visualize text to web page
def write_text(text):
    st.text(text)

# Function to visualize simple text to web page
def generic_write(text):
    st.write(text)

# Function to visualize dataframe to web page
def write_df(df):
    st.dataframe(df)

# Function to create a subheader to web page
def write_subheader(text):
    st.subheader(text)

# Function to create a title to web page
def write_title(text):
    st.title(text)

# Function to create a title to web page
def write_header(text):
    st.header(text)

# Function to write markdown code to web page
def write_md(text):
    st.markdown(text)

# Function to write caption to web page
def write_caption(text):
    st.caption(text)

#####################################################
# Main
#####################################################

# Create a dynamic page
#st.set_page_config(layout= 'centered')
#st.header('Online Retail Sales and Customer Data')
#st.write('Use the sidebar to choose what to visualize.')


# Initializing web page
st.set_page_config(layout= 'centered')
write_title('Online Retail')
st.markdown('SCRIVI IN BREVE DI COSA SI TRATTA IL PROGETTO')
st.markdown("**Source dataset**: [Kaggle page](https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data)")
write_md('For more information read ```README.md``` file')

st.sidebar.write('Settings')
write_md('> Use the **sidebar menu** to show advanced features')

#####################################################
## Data Exploration and Data Cleaning
#####################################################
st.sidebar.write('What do you want to see?')
if st.sidebar.checkbox('EDA'):
    st.title('Exploration Data Analysis')
    ###################
    # BEFORE CLEANING
    st.header('Before Cleaning')
    st.write('Online Retail dataframe:')
    st.write(df_sales)
    st.write('Rows and columns:', df_sales.shape)
    
    st.write('Dataframe head and tail:')
    st.write(df_sales.head(5))
    st.write(df_sales.tail(5))

    st.write('Some numerical informations:')
    st.write(df_sales.describe())
    
    ###################
    # AFTER CLEANING
    st.header('After Cleaning')
    #df_sales = cl.return_clean_dataset()
    st.write('Online Retail dataframe:')
    st.write(df_sales)
    st.write('Rows and columns:', df_sales.shape)
    
    st.write('Dataframe head and tail:')
    st.write(df_sales.head(5))
    st.write(df_sales.tail(5))

    st.write('Some numerical informations:')
    st.write(df_sales.describe())


#####################################################
## Data Visualization
#####################################################
if st.sidebar.checkbox('PLOTS'):
    st.title('Data Visualization')
    st.write('PLOTS')

#####################################################
# Machine Learning
#####################################################
if st.sidebar.checkbox('MODELS'):
    st.title('Machine Learning')
    st.write('Model')
