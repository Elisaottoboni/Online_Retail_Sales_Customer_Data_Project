import pandas as pd
import numpy as np

def return_clean_dataset():
    answer = input("Do you want to maintain the row with NaN value (yes/no): ").lower()
    if answer == "no":
        sales_df = pd.read_csv('delete_row.csv')
    elif answer == "yes":
        sales_df = pd.read_csv('change_value.csv')
    else:
        print('Wrong answer')
    return sales_df  


def delete_row():
    sales_df = pd.read_csv('online_retail.csv')
    
    sales_df.dropna(inplace = True)
    sales_df.reset_index(drop = True, inplace = True)
    return sales_df


def change_value():
    sales_df = pd.read_csv('online_retail.csv')
    
    sales_df.Description.fillna('NO DESCRIPTION', inplace = True)
    unique_CustomerID_mask = sales_df.CustomerID.unique()
    customerId_mask = sales_df.CustomerID.isna()
    InvoiceNo_without_CustomerID = sales_df.InvoiceNo[customerId_mask]
    new_CustomerID = generate_random_numbers(InvoiceNo_without_CustomerID.nunique(), unique_CustomerID_mask)
    unique_InvoiceNo_with_NaN = InvoiceNo_without_CustomerID.unique()
    for invoice in unique_InvoiceNo_with_NaN:
        # Randomly select one of the numbers from the array
        new_customerID = np.random.choice(new_CustomerID)
        # Assign the new CustomerID to the rows with the same InvoiceNo
        sales_df.loc[(sales_df['InvoiceNo'] == invoice) & customerId_mask, 'CustomerID'] = new_customerID
    return sales_df


def generate_random_numbers(length, existing_numbers):
    generated_numbers = []
    # Generate random numbers until the length of the array reaches the desired length
    while len(generated_numbers) < length:
        random_number = np.random.randint(10000, 20000)
        # Ensure that the randomly generated number is not already present in the existing_numbers array
        if random_number not in existing_numbers:
            generated_numbers.append(random_number)
    return generated_numbers


import tkinter as tk
from tkinter import simpledialog

# Funzione per visualizzare la finestra di dialogo di input al centro dello schermo
def mostra_input():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale

    # Ottieni le dimensioni dello schermo
    larghezza_schermo = root.winfo_screenwidth()
    altezza_schermo = root.winfo_screenheight()

    # Crea la finestra di dialogo di input
    input_utente = simpledialog.askstring("Input", "Inserisci il testo:")

    # Calcola le coordinate per centrare la finestra di dialogo
    x = larghezza_schermo // 2 - 150
    y = altezza_schermo // 2 - 50

    # Posiziona la finestra di dialogo al centro dello schermo
    root.geometry(f"300x100+{x}+{y}")

    # Mostra la finestra di dialogo di input
    root.mainloop()


