
import os
import pandas as pd
import numpy as np


class DataCleanerLoader:
    """
    Handles raw data loading, cleaning, interpolation, 
    hilly feature generation, and final export to clean CSV.
    """

    def __init__(self,
                 input_path="data/raw_data.csv",
                 output_path="artifacts/data/predict_clean_data.csv",
                 date_col="Date",
                 target_col="Average_Price",
                 supply_col="Supply_Volume",
                 ):

        self.input_path = input_path
        self.output_path = output_path
        self.date_col = date_col
        self.target_col = target_col
        self.supply_col = supply_col

        # Default remove list if none provided
        self.remove_cols =   [
            'imported_tomato_price',
            'Dhading_Wind_Speed', 
            'Dhading_Temperature',
            'Dhading_Precipitation',
            # 'Dhading_Rainfall_MM',
            # 'Dhading_Air_Pressure',
            'Kathmandu_Wind_Speed', 
            'Kathmandu_Precipitation', 
            # 'Kathmandu_Air_Pressure',
            'Kavre_Wind_Speed', 
            'Kavre_Temperature', 
            'Kavre_Precipitation',
            # 'Kavre_Rainfall_MM', 
            'Kavre_Air_Pressure',
            'Sarlahi_Wind_Speed', 
            'Sarlahi_Precipitation',
            # 'Sarlahi_Air_Pressure',
            'Hilly_Precipitation', 
            'Hilly_Rainfall_MM'
            # 'Hilly_Air_Pressure'
        ]
 

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    # -----------------------------
    # Load raw CSV
    # -----------------------------
    def load_raw(self):
        df = pd.read_csv(self.input_path)
        print(f"✅ Loaded raw data from {self.input_path} with shape {df.shape}")
        return df

    # -----------------------------
    # Basic formatting & cleanup
    # -----------------------------
    def basic_cleaning(self, df):
        df = df.copy()

        # Ensure proper date format and sorting
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
            df = df.sort_values(self.date_col).reset_index(drop=True)

        # Drop obvious garbage columns
        drop_these = [c for c in self.remove_cols if c in df.columns]
        if drop_these:
            df = df.drop(columns=drop_these)
            print(f"🧹 Dropped specified columns: {drop_these}")

        # Convert boolean columns → int
        bool_cols = df.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
            print(f"ℹ️ Converted boolean columns to int: {list(bool_cols)}")

        return df

    # -----------------------------
    # Interpolate supply volume
    # -----------------------------
    def interpolate_supply(self, df):
        if self.supply_col in df.columns:
            before_na = df[self.supply_col].isna().sum()
            df[self.supply_col] = df[self.supply_col].interpolate(method="linear", limit_direction="both")
            after_na = df[self.supply_col].isna().sum()
            print(f"📈 Interpolated '{self.supply_col}' — filled {before_na - after_na} missing values.")
        return df

    # -----------------------------
    # Create Hilly-region aggregate features
    # -----------------------------
    def add_hilly_features(self, df):
        """Create 'Hilly_Temperature' and 'Hilly_Rainfall_MM' 
        as averages across multiple hilly districts."""
        df = df.copy()

        df["Hilly_Temperature"] = df[["Dhading_Temperature", "Kavre_Temperature"]].mean(axis=1)
        df["Hilly_Precipitation"] = df[["Dhading_Precipitation", "Kavre_Precipitation"]].mean(axis=1)
        df["Hilly_Air_Pressure"] = df[["Dhading_Air_Pressure", "Kavre_Air_Pressure"]].mean(axis=1)
        df["Hilly_Rainfall_MM"] = df[["Dhading_Rainfall_MM", "Kavre_Rainfall_MM"]].mean(axis=1)


        

        # remove individual hilly district columns now that aggregates exist
        
        

        return df

    # -----------------------------
    # Final cleanup (drop missing target, reset)
    # -----------------------------
    def finalize(self, df):
        if self.target_col in df.columns:
            before = len(df)
            df = df.dropna(subset=[self.target_col])
            after = len(df)
            print(f"🧭 Dropped {before - after} rows missing target '{self.target_col}'.")

        df = df.reset_index(drop=True)
        return df

    # -----------------------------
    # Full cleaning pipeline
    # -----------------------------
    def clean_and_save(self):
        df = self.load_raw()
        df = self.interpolate_supply(df)
        df = self.add_hilly_features(df)
        df = self.basic_cleaning(df)
        df = self.finalize(df)

        df.to_csv(self.output_path, index=False)
        print(f"✅ Cleaned data saved to {self.output_path} (shape={df.shape})")

        return df
