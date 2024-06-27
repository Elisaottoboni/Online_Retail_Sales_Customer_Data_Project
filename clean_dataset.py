import pandas as pd
import numpy as np

def clean_dataset():
    df_sales = pd.read_csv('dataset/online_retail.csv')
    
    NaN_description_unique = df_sales[df_sales['Description'].isnull()]['StockCode'].unique()
    description_unique = df_sales[df_sales['Description'].notna()]['StockCode'].unique()
    for SC in NaN_description_unique:
        if SC in description_unique:
            description_mapping = df_sales.loc[df_sales['StockCode'] == SC, 'Description'].value_counts().idxmax()
            df_sales.loc[(df_sales['StockCode'] == SC) & (df_sales['Description'].isna()), 'Description'] = description_mapping
        else:
            df_sales.loc[df_sales['StockCode'] == SC, 'Description'] = 'NO DESCRIPTION'
    
    df_sales = df_sales.dropna()

    df_sales["InvoiceDate"] = pd.to_datetime(df_sales["InvoiceDate"])
    df_sales["Time"] = df_sales["InvoiceDate"].dt.time
    df_sales["Day"] = df_sales["InvoiceDate"].dt.day
    df_sales["Month"] = df_sales["InvoiceDate"].dt.month
    df_sales["Year"] = df_sales["InvoiceDate"].dt.year

    df_sales['TotalSales'] = df_sales['Quantity'] * df_sales['UnitPrice']

    df_sales['Season'] = df_sales.Month.apply(get_month)

    df_sales = df_sales[df_sales['Quantity'] > 0]

    df_sales.to_csv(f'{'dataset'}/{'clean_dataset.csv'}', index=False)
    return df_sales


def get_month(month):
  if month in [3, 4, 5]:
    return 'Spring'
  elif month in [6, 7, 8]:
    return 'Summer'
  elif month in [9, 10, 11]:
    return 'Autumn'
  elif month in [12, 1, 2]:
    return 'Winter'
  else:
    return 'Unknown'

clean_dataset()


