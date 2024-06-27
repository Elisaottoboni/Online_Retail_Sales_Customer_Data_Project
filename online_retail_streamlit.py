# python3 -m streamlit run online_retail_streamlit.py

#####################################################
# Libraries
#####################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from prettytable import PrettyTable


#####################################################
# Reads datasets
#####################################################

df_sales = pd.read_csv('dataset/online_retail.csv')

#####################################################
# Functions
#####################################################
# Functions for the plot of the outlier
def box_and_scatter_plot(df, df_column):
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))

    # Create a box plot in the first subplot
    sns.boxplot(x=df[df_column], ax = axs[0])
    axs[0].set_xlabel(df_column)
    axs[0].set_title('Box Plot of ' + df_column + ' to Identify Outliers')

    # Create a scatter plot in the second subplot
    sns.scatterplot(x = df.index, y = df[df_column], ax=axs[1])
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(df_column)
    axs[1].set_title('Scatter Plot of ' + df_column + ' to Identify Outliers')

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()


# Function to encod the object valu of the dataset
def encoding_cl(df, df_col):
  label_encoder = LabelEncoder()
  for column in df_col:
    if df[column].dtype == 'object':
      df[column] = label_encoder.fit_transform(df[column])
  return df

# 
def prepare_data(df, target, columns_to_exclude=[]):
    # Drop 'time' column if present
    if 'InvoiceDate' in df.columns:
        df = df.drop(columns=['InvoiceDate'])

    # Dynamically select features by excluding the columns we don't need
    features = [col for col in df.columns if col not in columns_to_exclude + [target]]

    X = df[features]
    y = df[target]

    return X, y


def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model


def train_and_evaluate(df, target, model, columns_to_exclude=[]):
    X, y = prepare_data(df, target, columns_to_exclude)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print the shape of our train and testing sets
    table_shape = PrettyTable()
    table_shape.field_names = ["", "X_shape", "y_shape"]
    table_shape.add_row(["Training set shape", X_train.shape, y_train.shape])
    table_shape.add_row(["Testing set shape", X_test.shape, y_test.shape])
    print(table_shape)

    model_train = train_model(X_train, y_train, model)

    evaluate_and_plot(X_test, y_test, model_train, target)


def evaluate_and_plot(X_test, y_test, model, target):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2_Score = r2_score(y_test, y_pred)
    # intercept = model.intercept_

    results = PrettyTable()
    results.field_names = ["Metric", "Value"]
    results.add_row(["Mean Squared Error", MSE])
    results.add_row(["Mean Absolute Error", MAE])
    results.add_row(["R2 Score", R2_Score])
    # results.add_row(["Intercept", intercept])
    print(results)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel(target)
    plt.title(f'Actual vs Predicted {target}')
    plt.legend()
    plt.show()


def train_and_evaluate_polynomial(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    table_shape = PrettyTable()
    table_shape.field_names = ["", "X_shape", "y_shape"]
    table_shape.add_row(["Training set shape", X_train.shape, y_train.shape])
    table_shape.add_row(["Testing set shape", X_test.shape, y_test.shape])
    print(table_shape)

    trained_model = train_model(X_train, y_train, model)
    evaluate_and_plot(X_test, y_test, trained_model, target)
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
    df_sales = cld.clean_dataset()
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
