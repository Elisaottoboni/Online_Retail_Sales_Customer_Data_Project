import pandas as pd
import numpy as np

def return_clean_dataset():
    answer = input("Do you want to maintain the row with NaN value (yes/no): ").lower()
    if answer == "no":
        df_sales = pd.read_csv('delete_row.csv')
    elif answer == "yes":
        answer = input("Press 1 if you want to delete CustomerId column"
                        "\n Press 2 if you want to assign random number to CustomerID"
                        "\n Press 3 if you want to replace NaN value with '00000' value: ").lower()
        if answer == 1:
            df_sales = pd.read_csv('erased_CustomerID.csv')
        elif answer == 2:
            df_sales = pd.read_csv('change_random_value.csv')
        elif answer == 3:
            df_sales = pd.read_csv('change_00000_value.csv')
        else:
            print('Wrong answer')
    else:
        print('Wrong answer')
    return df_sales  


def delete_row():
    df_sales = pd.read_csv('online_retail.csv')
    
    df_sales.dropna(inplace = True)
    df_sales.to_csv('delete_row.csv', index=False)


def change_description():
    df_sales = pd.read_csv('online_retail.csv')
    
    NaN_description_unique = df_sales[df_sales['Description'].isnull()]['StockCode'].unique()
    description_unique = df_sales[df_sales['Description'].notna()]['StockCode'].unique()
    for SC in NaN_description_unique:
        if SC in description_unique:
            description_mapping = df_sales.loc[df_sales['StockCode'] == SC, 'Description'].value_counts().idxmax()
            df_sales.loc[(df_sales['StockCode'] == SC) & (df_sales['Description'].isna()), 'Description'] = description_mapping
        else:
            df_sales.loc[df_sales['StockCode'] == SC, 'Description'] = 'NO DESCRIPTION'
    return df_sales


def erased_CustomerID():
    df_sales = change_description()
    
    df_sales.drop(columns = ['CustomerID'], inplace = True)
    df_sales.to_csv('erased_CustomerID.csv', index = False)


def generate_random_numbers(length, existing_numbers):
    generated_numbers = []
    while len(generated_numbers) < length:
        random_number = np.random.randint(10000, 20000)
        if random_number not in existing_numbers:
            generated_numbers.append(random_number)
    return generated_numbers


def change_random_value():
    df_sales = change_description()
    
    unique_CustomerID_mask = df_sales.CustomerID.unique()
    customerId_mask = df_sales.CustomerID.isna()
    InvoiceNo_without_CustomerID = df_sales.InvoiceNo[customerId_mask]
    new_CustomerID = generate_random_numbers(InvoiceNo_without_CustomerID.nunique(), unique_CustomerID_mask)
    unique_InvoiceNo_with_NaN = InvoiceNo_without_CustomerID.unique()
    for invoice in unique_InvoiceNo_with_NaN:
        new_customerID = np.random.choice(new_CustomerID)
        df_sales.loc[(df_sales['InvoiceNo'] == invoice) & customerId_mask, 'CustomerID'] = new_customerID
    df_sales.to_csv('change_random_value.csv', index = False)



def change_00000_value():
    df_sales = change_description()
    
    df_sales.loc[df_sales['CustomerID'].isna(), 'CustomerID'] = 00000.0
    df_sales.to_csv('change_00000_value.csv', index = False)