import zipfile,os
from pathlib import Path
from fitparse import FitFile
import pandas as pd
import numpy as np

if not os.path.isdir('runs'):
    os.mkdir('runs')
    for file in os.listdir(os.fsdecode('raw_fits_zipped')):
        filename = os.fsdecode(file)
        with zipfile.ZipFile('raw_fits_zipped/'+filename, 'r') as zip_ref:
            zip_ref.extractall('runs')

if not (Path('downhill.pkl').is_file()&Path('uphill.pkl').is_file()):
    dataframes = []
    for file in os.listdir(os.fsdecode('runs')):
        filename = os.fsdecode(file)
        rows = []

        fitfile = FitFile('runs/'+file)
        for record in fitfile.get_messages("record"):
            data = {}
            for field in record:
                data[field.name] = field.value
            rows.append(data)
        df = pd.DataFrame(rows)
        df = df[["distance", "enhanced_altitude", "enhanced_speed", "vertical_oscillation"]]
        df = df.dropna()
        df = df[df["distance"].diff()>0.1]
        df = df.reset_index(drop=True)
        df["altitude_smooth"] = df["enhanced_altitude"].rolling(window=5, center=True).mean()
        df["gradient"] = np.gradient(df["altitude_smooth"], df["distance"])
        df = df[(df["gradient"]>-0.5) & (df["gradient"]<0.5)]
        df = df[(df["gradient"].abs()>=0.02)]
        df["gradient_smooth"] = df["gradient"].rolling(window=5, center=True).mean()
        df = df.dropna(subset=["gradient_smooth"])
        df["roc_gradient"] = np.gradient(df["gradient_smooth"], df["distance"])
        dataframes.append(df)
        if filename=='19471318480_ACTIVITY.fit':
            df.to_pickle('test.pkl')
        print(f"{filename} was processed")

    combined_df = pd.concat(dataframes, ignore_index=True)


    uphill_df = combined_df[combined_df["gradient"]>0]
    downhill_df = combined_df[combined_df["gradient"]<0]

    downhill_df.to_pickle('downhill.pkl')
    uphill_df.to_pickle('uphill.pkl')
    combined_df.to_pickle('combined.pkl')
else:
    downhill_df = pd.read_pickle('downhill.pkl')
    uphill_df = pd.read_pickle('uphill.pkl')

print(f"The uphill dataframe consists of {len(uphill_df)} rows")
print(uphill_df.head())
print(f"The downhill dataframe consists of {len(downhill_df)} rows")
print(downhill_df.head())