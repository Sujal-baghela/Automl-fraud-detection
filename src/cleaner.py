import pandas as pd
import numpy as np
import logging


# FIX: removed unused 'from pyparsing import col' — pyparsing is not a standard
# ML dependency and was never used anywhere in this class


class DataCleaner:
    def __init__(self):
        pass

    def detect_missing(self, df):
        missing = df.isnull().sum()
        total_missing = missing.sum()

        logging.info(f"Total missing values found: {total_missing}")

        return missing

    def impute_missing(self, df):
        for col in df.columns:
            if df[col].isnull().sum() > 0:

                if df[col].dtype in ['int64', 'float64']:
                    mean_value = df[col].mean()
                    df[col] = df[col].fillna(mean_value)
                    logging.info(f"Filled missing numerical values in '{col}' with mean")

                else:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                    logging.info(f"Filled missing categorical values in '{col}' with mode")

        return df

    def remove_duplicates(self, df):
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]

        logging.info(f"{before - after} duplicate rows removed")

        return df

    def detect_outliers_iqr(self, df):
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

            if outliers > 0:
                logging.info(f"{outliers} outliers detected in column '{col}'")

        return df

    def clean(self, df):
        """
        Master cleaning method — run the full cleaning pipeline in correct order.

        Usage in train.py (call BEFORE train/test split to avoid leakage):
            cleaner = DataCleaner()
            df = cleaner.clean(df)

        Steps:
            1. Remove duplicate rows
            2. Impute missing values
            3. Log outlier counts (does not remove — tree models are robust to outliers)
        """
        logging.info("Starting data cleaning pipeline...")

        df = self.remove_duplicates(df)
        df = self.impute_missing(df)
        self.detect_outliers_iqr(df)   # log only, not removed

        logging.info("Data cleaning completed.")

        return df
