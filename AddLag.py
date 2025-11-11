

import pandas as pd
import numpy as np
import json
import os


class FeatureEngineer:
    def __init__(self,
                 lag_config_path="artifacts/configs/lag_config.json",
                 roll_config_path="artifacts/configs/roll_config.json"):
        self.lag_config_path = lag_config_path
        self.roll_config_path = roll_config_path
        self.lag_config = self._load_json(lag_config_path)
        self.roll_config = self._load_json(roll_config_path)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            print(f"⚠️ Config not found: {path}")
            return {}

    # -----------------------------
    # Apply lag features
    # -----------------------------
    def add_lag_features(self, df):
        df = df.copy()
        for col, lags in self.lag_config.items():
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        print(f"🕒 Added lag features for: {list(self.lag_config.keys())}")
        return df

    # -----------------------------
    # Apply rolling mean features
    # -----------------------------
    def add_rolling_features(self, df):
        df = df.copy()
        for col, windows in self.roll_config.items():
            if col not in df.columns:
                continue
            for w in windows:
                df[f"{col}_roll{w}"] = df[col].rolling(window=w).mean()
        print(f"📊 Added rolling features for: {list(self.roll_config.keys())}")
        return df

    # -----------------------------
    # Main method to apply all
    # -----------------------------
    def apply_features(self, df, use_lags=True):
        if not use_lags:
            print("🚫 Lag/Roll feature generation skipped.")
            return df

        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)

        # drop rows with NaN after shifting
        df = df.dropna().reset_index(drop=True)
        print(f"✅ Lag/Roll features added. Final shape: {df.shape}")
        return df
