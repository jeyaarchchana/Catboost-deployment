import pandas as pd

def extract_and_clean_city_features(df):
    df = df.copy()

    df['source_city'] = df['source_name'].str.split(" ", n=1, expand=True)[0].str.split("-", n=1, expand=True)[0]
    df["destination_city"] = df["destination_name"].str.split(" ", n=1, expand=True)[0].str.split("_", n=1, expand=True)[0]

    replace_map = {
        "del": "Delhi",
        "Bangalore": "Bengaluru",
        "AMD": "Ahmedabad",
        "Amdavad": "Ahmedabad"
    }

    df["source_city"] = df["source_city"].replace(replace_map)
    df["destination_city"] = df["destination_city"].replace(replace_map)

    return df
