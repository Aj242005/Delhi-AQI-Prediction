"""
build_dataset.py - reproducible rebuild of the training dataset.

Reproduces the four notebook stages as one command:

    python build_dataset.py

Inputs (Dataset/):
    original_data.csv                  raw OpenCity/CPCB Anand Vihar AQI table
    anand_vihar_weather_2017_2023.csv  cached Open-Meteo hourly weather
                                       (downloaded automatically if missing)

Outputs (Dataset/):
    cleaned_data.csv
    final_clean_aqi.csv
    final_clean_aqi_v3.csv
    ml_ready_data.csv                  <- what server.py and the training notebook use

The raw AQI file is the "Anand Vihar DPCC AQI Data 2017-2023" resource from
https://data.opencity.in/dataset/delhi-hourly-air-quality-reports
"""
import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
RAW_PATH = os.path.join(DATA_DIR, "original_data.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
FINAL_PATH = os.path.join(DATA_DIR, "final_clean_aqi.csv")
FINAL_V3_PATH = os.path.join(DATA_DIR, "final_clean_aqi_v3.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "anand_vihar_weather_2017_2023.csv")
ML_READY_PATH = os.path.join(DATA_DIR, "ml_ready_data.csv")

LAT, LON = 28.6469, 77.3160
WEATHER_START = "2017-10-01"
WEATHER_END = "2023-12-31"
REALISTIC_FLOOR = 40


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------
# Stage 0 - unpivot the raw monthly CPCB workbook
# ---------------------------------------------------------
def clean_raw():
    print("\n=== STAGE 0: CLEANING RAW AQI TABLE ===")
    df = pd.read_csv(RAW_PATH, header=None, names=range(26), dtype=str)
    df[0] = df[0].str.strip()
    df = df.dropna(how="all")
    df = df[df[0] != ""]
    df = df[df[0] != "Year"]

    df["Month_Year"] = np.where(df[0].str.contains("-", na=False), df[0], np.nan)
    df["Month_Year"] = df["Month_Year"].ffill()
    df = df[df[0].str.isnumeric() == True]
    df = df.drop(columns=[25])

    df = df.rename(columns={0: "Day"})
    hours = [f"{str(i).zfill(2)}:00:00" for i in range(24)]
    df = df.rename(columns={i + 1: hours[i] for i in range(24)})
    df = df[["Month_Year", "Day"] + hours]

    df_melted = pd.melt(df, id_vars=["Month_Year", "Day"], value_vars=hours,
                        var_name="Time", value_name="AQI")
    df_melted["Datetime"] = pd.to_datetime(
        df_melted["Month_Year"] + "-" + df_melted["Day"] + " " + df_melted["Time"],
        format="%B-%Y-%d %H:%M:%S",
        errors="coerce",
    )
    df_melted = df_melted.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df_melted["AQI"] = pd.to_numeric(df_melted["AQI"], errors="coerce")
    df_melted = df_melted[["Datetime", "Month_Year", "Day", "Time", "AQI"]]
    df_melted.to_csv(CLEANED_PATH, index=False)
    print(f"Rows: {len(df_melted)} | missing AQI: {df_melted['AQI'].isna().sum()}")
    return df_melted


# ---------------------------------------------------------
# Stage 1 - hourly timeline + layered missing-value strategy
# ---------------------------------------------------------
def fill_missing():
    print("\n=== STAGE 1: MISSING-VALUE HANDLING ===")
    df = pd.read_csv(CLEANED_PATH, parse_dates=["Datetime"])
    first_reading_row = df["AQI"].first_valid_index()
    print(f"Original rows: {len(df)} | sensor first recording: {df.loc[first_reading_row, 'Datetime']}")
    df = df.loc[first_reading_row:].reset_index(drop=True)

    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    full_index = pd.date_range(start=df["Datetime"].min(), end=df["Datetime"].max(), freq="h")
    df = df.set_index("Datetime").reindex(full_index).reset_index()
    df = df.rename(columns={"index": "Datetime"})
    df["Month_Year"] = df["Datetime"].dt.strftime("%B-%Y")
    df["Day"] = df["Datetime"].dt.day
    df["Time"] = df["Datetime"].dt.strftime("%H:%M:%S")
    print(f"Rows after timeline repair: {len(df)} | missing AQI: {df['AQI'].isna().sum()}")

    df["AQI_Clean"] = df["AQI"].ffill(limit=6)
    df["AQI_Clean"] = df["AQI_Clean"].interpolate(method="linear", limit=24, limit_direction="both")
    df["YearMonth"] = df["Datetime"].dt.to_period("M")
    df["AQI_Clean"] = df["AQI_Clean"].fillna(df.groupby("YearMonth")["AQI_Clean"].transform("mean"))
    df["AQI_Clean"] = df["AQI_Clean"].fillna(
        df.groupby(df["Datetime"].dt.month)["AQI_Clean"].transform("mean"))
    df["AQI_Clean"] = df["AQI_Clean"].fillna(df["AQI_Clean"].mean())

    df = df.drop(columns=["AQI", "YearMonth"]).rename(columns={"AQI_Clean": "AQI"})
    fault_mask = df["AQI"] < REALISTIC_FLOOR
    print(f"Flagging {fault_mask.sum()} sensor-fault rows below the realistic floor")
    df.loc[fault_mask, "AQI"] = np.nan
    df["AQI"] = df["AQI"].ffill(limit=6)
    df["AQI"] = df["AQI"].interpolate(method="linear", limit=24, limit_direction="both")
    df["YearMonth"] = df["Datetime"].dt.to_period("M")
    df["AQI"] = df["AQI"].fillna(df.groupby("YearMonth")["AQI"].transform("mean"))
    df["AQI"] = df["AQI"].fillna(df.groupby(df["Datetime"].dt.month)["AQI"].transform("mean"))
    df["AQI"] = df["AQI"].fillna(df["AQI"].mean())
    df = df.drop(columns=["YearMonth"])

    print(f"Rows: {len(df)} | missing AQI: {df['AQI'].isna().sum()} | "
          f"range: {df['Datetime'].min()} -> {df['Datetime'].max()}")
    df.to_csv(FINAL_PATH, index=False)
    return df


# ---------------------------------------------------------
# Stage 2 - repair the 720 missing hourly timestamps + is_real flag
# ---------------------------------------------------------
def repair_timeline_and_flag(df_final):
    print("\n=== STAGE 2: TIMELINE REPAIR + is_real FLAG ===")
    df_clean = pd.read_csv(CLEANED_PATH, parse_dates=["Datetime"])
    df_final = df_final.copy()

    full_timeline = pd.date_range(start=df_final["Datetime"].min(), end=df_final["Datetime"].max(), freq="h")
    df_final = df_final.set_index("Datetime").reindex(full_timeline).rename_axis("Datetime").reset_index()
    df_final["AQI"] = df_final["AQI"].interpolate(method="linear", limit_direction="both")

    merged = pd.merge(df_final[["Datetime", "AQI"]], df_clean[["Datetime", "AQI"]],
                      on="Datetime", how="left", suffixes=("_final", "_original"))
    df_final["is_real"] = (~merged["AQI_original"].isna()) & (merged["AQI_original"] >= REALISTIC_FLOOR)

    missing = len(full_timeline.difference(df_final["Datetime"]))
    print(f"Rows: {len(df_final)} | missing timestamps: {missing} | "
          f"real readings: {int(df_final['is_real'].sum())}")
    df_final.to_csv(FINAL_V3_PATH, index=False)
    return df_final


# ---------------------------------------------------------
# Stage 3 - weather merge + features + targets
# ---------------------------------------------------------
def download_weather():
    if os.path.exists(WEATHER_PATH):
        print("Using cached weather file")
        return
    print("Downloading Open-Meteo hourly weather...")
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={LAT}&longitude={LON}&start_date={WEATHER_START}&end_date={WEATHER_END}"
           "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=Asia%2FKolkata")
    req = urllib.request.Request(url, headers={"User-Agent": "anand-vihar-aqi-project/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = __import__("json").loads(r.read().decode("utf-8"))
    df_weather = pd.DataFrame({
        "Datetime": pd.to_datetime(data["hourly"]["time"]),
        "Temp_C": data["hourly"]["temperature_2m"],
        "Humidity": data["hourly"]["relative_humidity_2m"],
        "Wind_Speed": data["hourly"]["wind_speed_10m"],
    })
    df_weather.to_csv(WEATHER_PATH, index=False)
    print(f"Saved weather: {len(df_weather)} rows")


def build_ml_ready(df_final):
    print("\n=== STAGE 3: WEATHER MERGE + FEATURE ENGINEERING ===")
    download_weather()
    df_weather = pd.read_csv(WEATHER_PATH, parse_dates=["Datetime"])

    df_ml = pd.merge(df_final[["Datetime", "AQI", "is_real"]], df_weather,
                     on="Datetime", how="left")
    df_ml[["Temp_C", "Humidity", "Wind_Speed"]] = df_ml[["Temp_C", "Humidity", "Wind_Speed"]].ffill().bfill()

    df_ml["Hour"] = df_ml["Datetime"].dt.hour
    df_ml["Month"] = df_ml["Datetime"].dt.month
    df_ml["DayOfWeek"] = df_ml["Datetime"].dt.dayofweek
    df_ml["Is_Weekend"] = (df_ml["DayOfWeek"] >= 5).astype(int)

    df_ml["AQI_Lag_1"] = df_ml["AQI"].shift(1)
    df_ml["AQI_Lag_3"] = df_ml["AQI"].shift(3)
    df_ml["AQI_Lag_6"] = df_ml["AQI"].shift(6)
    df_ml["AQI_Lag_24"] = df_ml["AQI"].shift(24)
    df_ml["AQI_Lag_48"] = df_ml["AQI"].shift(48)
    df_ml["Rolling_Mean_6"] = df_ml["AQI"].rolling(window=6).mean()
    df_ml["Rolling_Mean_24"] = df_ml["AQI"].rolling(window=24).mean()
    df_ml["Rolling_Std_24"] = df_ml["AQI"].rolling(window=24).std()
    df_ml["AQI_Diff_24"] = df_ml["AQI"] - df_ml["AQI_Lag_24"]
    df_ml["Target_AQI_24h"] = df_ml["AQI"].shift(-24)
    df_ml["Target_Is_Real_24h"] = df_ml["is_real"].shift(-24)

    df_ml = df_ml.dropna().reset_index(drop=True)
    df_ml.to_csv(ML_READY_PATH, index=False)
    print(f"Rows: {len(df_ml)} | columns: {len(df_ml.columns)}")
    print(f"sha256: {sha256(ML_READY_PATH)}")
    return df_ml


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for path in (RAW_PATH,):
        if not os.path.exists(path):
            raise SystemExit(f"Missing raw input: {path}\nSee module docstring for the source URL.")
    clean_raw()
    df_final = fill_missing()
    df_final_v3 = repair_timeline_and_flag(df_final)
    build_ml_ready(df_final_v3)
    print("\nDataset rebuild complete.")


if __name__ == "__main__":
    main()
