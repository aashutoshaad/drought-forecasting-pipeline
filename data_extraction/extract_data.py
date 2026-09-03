import pandas as pd
import os

os.makedirs('data', exist_ok=True)

xl = pd.ExcelFile('Data_spei1_30_60_90_suj.xlsx')

for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    filename = f"data/{sheet_name.replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {sheet_name} to {filename}")
