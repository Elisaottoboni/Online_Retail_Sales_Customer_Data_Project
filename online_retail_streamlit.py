# python3 -m streamlit run online_retail_streamlit.py

#####################################################
# Libraries
#####################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


#####################################################
# Reads datasets
#####################################################

df_sales = pd.read_csv('dataset/online_retail.csv')
df_sales_clean = pd.read_csv('dataset/clean_dataset.csv')

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
    st.pyplot(fig)


# Function to encod the object valu of the dataset
def encoding_cl(df, df_col):
    label_encoder = LabelEncoder()
    for column in df_col:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
    return df

# 
def prepare_data(df, target, selected_columns=[]):
    # Drop 'time' column if present
    if 'InvoiceDate' in df.columns:
        df = df.drop(columns=['InvoiceDate'])

    # Dynamically select features by excluding the columns we don't need
    # features = [col for col in df.columns if col not in columns_to_exclude + [target]]
    features = [col for col in selected_columns if col != target]

    X = df[features]
    y = df[target]

    return X, y

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def train_and_evaluate(df, target, model, columns_to_exclude=[]):
    X, y = prepare_data(df, target, columns_to_exclude)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    table_shape = pd.DataFrame({
    "": ["Training set shape", "Testing set shape"],
    "X_shape": [X_train.shape, X_test.shape],
    "y_shape": [y_train.shape, y_test.shape]
    })
    st.table(table_shape)

    model_train = train_model(X_train, y_train, model)

    evaluate_and_plot(X_test, y_test, model_train, target)

def evaluate_and_plot(X_test, y_test, model, target):
    y_pred = model.predict(X_test)

    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2_Score = r2_score(y_test, y_pred)

    results = pd.DataFrame({
    "Metric": ["Mean Squared Error", "Mean Absolute Error", "R2 Score"],
    "Value": [MSE, MAE, R2_Score]
    })
    st.table(results)


def imp_images(name):
    st.image(f'images/{name}', use_column_width = True)


def imp_images_models(target, model):
    st.image(f'images/Actual_Predicted_{target}_with_{model}.png', caption = f'Actual Predicted and Residuals {target} with {model}', use_column_width = True)

#####################################################
# Main
#####################################################

# Initializing web page
st.set_page_config(layout= 'centered')
st.title('Online Retail Sales and Customer Data')
st.markdown("""
            This dataset provides comprehensive insights into online retail operations, making it a valuable resource for various research and business applications. 
            Its columns can be split into:
            - **InvoiceNo**: Unique identifiers for transactions, aiding in sales/purchase identification and treasury management.
            - **InvoiceDate**: Date-time stamps for transactions, useful for analyzing purchasing patterns and record-keeping.
            - **StockCode**: Alphanumeric codes for each item, facilitating seamless inventory management.
            - **Description**: Brief explanations of products, enhancing customer understanding and decision-making.
            - **Quantity**: Logs of units sold per transaction, crucial for cost calculations and inventory tracking.
            - **UnitPrice**: Prices per unit sold, essential for revenue calculations and pricing strategies.
            - **Country**: Records of transaction locations, supporting customer segmentation and regional performance analysis.
            - **Dataset**: Named online_retail.csv, sortable and valuable for studying online sales trends, customer profiling, and inventory management strategies.
            """)
st.markdown("**Source dataset**: [Kaggle page](https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data)")

st.sidebar.write('Settings')
st.markdown('> Use the **sidebar menu** to show advanced features')

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
    df_sales = df_sales_clean
    st.write('Online Retail dataframe:')
    st.write(df_sales)
    st.write('Rows and columns:', df_sales.shape)
    
    st.write('Dataframe head and tail:')
    st.write(df_sales.head(5))
    st.write(df_sales.tail(5))

    st.write('Some numerical informations:')
    st.write(df_sales.describe())
    
    st.subheader('Outliers')
    box_and_scatter_plot(df_sales, 'Quantity')
    st.write('The values regarded as outliers are:')
    st.write(df_sales[df_sales['Quantity'] >= 10000])
    st.write('As these values are very far from the average trend, it is a good idea to eliminate them in order to make a better analysis later on.')
    quantity_mask = df_sales['Quantity'] < 10000
    df_sales = df_sales[quantity_mask]
    box_and_scatter_plot(df_sales, 'Quantity')
    
    box_and_scatter_plot(df_sales, 'UnitPrice')


#####################################################
## Data Visualization
#####################################################
if st.sidebar.checkbox('PLOTS'):
    st.title('Data Visualization')
    
    colors = ['#FFB6C1', '#FFDAB9', '#ADD8E6', '#98FB98', '#FFA07A', '#87CEFA', '#FF69B4', '#F0E68C', '#D3D3D3', '#B0C4DE']
    
    st.subheader('Sales by Country')
    imp_images('Sales_Country.png')
    st.subheader('Sales by Country and Top 10 Country by Sales')
    imp_images('Top_10_Sales.png')
    st.subheader('Top 10 Best Selling Products')
    imp_images('Top_10_Products.png')
    st.subheader('Top 10 Best Selling Products Distributed by Country')
    imp_images('Top_10_Products_by_Country.png')
    st.subheader('Top 5 products sold per Country')
    list_nation = df_sales['Country'].unique()
    choice = st.selectbox("Select a Country", list_nation)
    st.image(f'images/sales_distribution_{choice}.png', use_column_width = True)
    
    st.subheader('Unit Price for each product')
    list_product = df_sales['Description'].unique()
    # min_date = df_sales['Date'].min()
    # max_date = df_sales['Date'].max()
    # start_date, end_date = st.slider("Select a range of dates", min_date, max_date, value=(min_date, max_date))
    
    start_date = '2010-12-01'
    end_date = '2011-12-09'
    choice_p = st.selectbox("Select a product", list_product)
    specific_product = df_sales[df_sales['Description'] == choice_p]
    specific_product = specific_product[(specific_product['Date'] >= start_date) & (specific_product['Date'] <= end_date)]
    daily_sales = specific_product.groupby(['Date', 'Country'])['UnitPrice'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    countries = daily_sales['Country'].unique()
    for country in countries:
        data_country = daily_sales[daily_sales['Country'] == country]
        ax.plot(data_country['Date'], data_country['UnitPrice'], 'o-', label=country)

    ax.set_xlabel('Date')
    ax.set_ylabel('Unit Price')
    ax.set_title(f'Price of {choice_p} change over time by country ({start_date} to {end_date})')
    ax.legend()
    st.pyplot(fig)

#####################################################
# Machine Learning
#####################################################
if st.sidebar.checkbox('MODELS'):
    st.title('Machine Learning')
    st.subheader('Dataset')
    df_sales1 = df_sales.copy()
    st.write(df_sales1)
    df_col = df_sales1.columns
    df_sales1 = encoding_cl(df_sales1, df_col)
    st.subheader('Correlation Matrix')
    st.write("Dataframe dopo l'encoding:")
    st.write(df_sales1)

    imp_images('Correlation_matrix.png')

    st.subheader('Models')
    model_dict = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    list_of_models = list(model_dict.keys())
    choice_m = st.selectbox("Select a model", list_of_models)
    model = model_dict[choice_m]

    list_of_targets = ['TotalSales', 'UnitPrice']
    choice_t = st.selectbox("Select a target", list_of_targets)
    target = choice_t
    
    columns = [col for col in df_sales1.columns if col not in ['InvoiceDate', 'index']]
    selected_columns = st.multiselect("Select columns to include", columns, default=columns)

    if st.button("Model evaluation"):
        if len(selected_columns) < 2 or target not in selected_columns:
            st.error("Please select at least one feature column and the target column.")
        else:
            train_and_evaluate(df_sales1, target, model, selected_columns)
            if len(selected_columns) == len(columns):
                imp_images_models(target, choice_m)