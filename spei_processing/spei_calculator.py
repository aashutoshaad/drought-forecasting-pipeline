import pandas as pd
import numpy as np
from math import sin, tan, acos, radians, pi
from scipy.stats import pearson3, norm
import os

def day_length_correction(lat, month):
    lat = radians(lat)
    delta = 23.45 * sin(radians(360 * (284 + month) / 365))
    delta = radians(delta)
    ws = acos(-tan(lat) * tan(delta))
    N = 24 / pi * ws
    return N / 12

def thornthwaite_pet(monthly_temp, lat, months, rolling_window=12):
    PET = []
    monthly_temp_pos = monthly_temp.clip(lower=0)
    rolling_I = ((monthly_temp_pos / 5.0) ** 1.514).rolling(
        window=rolling_window, min_periods=1
    ).sum()

    for i, (T, m) in enumerate(zip(monthly_temp, months)):
        I = rolling_I.iloc[i]
        if I == 0 or T <= 0:
            pet_val = 0
        else:
            a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
            K = day_length_correction(lat, m)
            pet_val = 16 * K * ((10 * T / I) ** a)
        PET.append(pet_val)
    return pd.Series(PET, index=monthly_temp.index)

def spei_rolling(series, scale=1):
    rolling_sum = series.rolling(window=scale, min_periods=scale).sum().dropna()
    if len(rolling_sum) < 5:
        return pd.Series([np.nan] * len(series), index=series.index)
    params = pearson3.fit(rolling_sum)
    cdf = pearson3.cdf(rolling_sum, *params)
    spei_values = pd.Series(norm.ppf(cdf), index=rolling_sum.index)
    result = pd.Series(np.nan, index=series.index)
    result.loc[spei_values.index] = spei_values
    return result


base_dir = r"C:\Users\Acer\Downloads\python\Spei_All_Included"
file_path = os.path.join(base_dir, "monthly_data.xlsx")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found! Please check path: {file_path}")

all_sheets = pd.read_excel(file_path, sheet_name=None)
output_sheets = {}

scales = [1, 3, 6, 9, 12]  # SPEI scales
steps = [1, 2, 3]  # 30/60/90 day intervals, in months

for station_name, df in all_sheets.items():
    print(f"Processing station: {station_name}")

    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    lat = df["Latitude"].iloc[0]

    df["PET"] = thornthwaite_pet(df["Tmean"], lat, df["Month"], rolling_window=12)
    df["WB"] = df["Precipitation"] - df["PET"]

    for scale in scales:
        for step in steps:
            col_name = f"SPEI{scale}_{step*30}d"
            if step == 1:
                df[col_name] = spei_rolling(df["WB"], scale=scale)
            else:
                df[col_name] = spei_rolling(df["WB"].rolling(window=step).sum(), scale=scale)

    output_sheets[station_name] = df

os.makedirs(base_dir, exist_ok=True)
out_path = os.path.join(base_dir, "output_all_spei_30_60_90.xlsx")

if os.path.exists(out_path):
    i = 1
    while True:
        alt_path = out_path.replace(".xlsx", f"_v{i}.xlsx")
        if not os.path.exists(alt_path):
            out_path = alt_path
            break
        i += 1

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for station_name, df in output_sheets.items():
        df.to_excel(writer, sheet_name=station_name[:31], index=False)

print(f"Done. All SPEI scales (1,3,6,9,12) for 30/60/90 days saved: {out_path}")
