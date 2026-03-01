import pandas as pd
import os
import logging


# FIX: logger is already correctly defined here as a module-level logger
# The problem was that all methods below were calling logging.xxx() directly
# which uses the ROOT logger — and since train.py also configures the root
# logger via basicConfig(), every message was firing TWICE.
# Fix: replace every logging.xxx() with logger.xxx() throughout the file.
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load CSV file into pandas DataFrame
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)

        if file_size_mb > 50:
            logger.warning(f"Large file detected: {file_size_mb:.2f} MB")  # FIX

        df = pd.read_csv(self.file_path)
        logger.info(f"Dataset loaded successfully with shape {df.shape}")   # FIX

        return df

    def get_metadata(self, df):
        """
        Generate dataset metadata summary
        """
        metadata = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numerical_features": df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist(),
            "categorical_features": df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
        }

        logger.info("Metadata generated successfully")   # FIX
        return metadata

    def split_features_target(self, df, target_column):
        """
        Split dataset into features and target
        """
        if target_column not in df.columns:
            raise ValueError("Target column not found in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(                                     # FIX
            f"Features and target split completed. Target: {target_column}"
        )

        # Log Class Distribution
        class_counts = y.value_counts()
        logger.info(f"Class Distribution:\n{class_counts}")   # FIX

        # Detect Imbalance
        if len(class_counts) == 2:
            minority_ratio = class_counts.min() / class_counts.sum()

            if minority_ratio < 0.1:
                logger.warning(                               # FIX
                    f"High class imbalance detected. "
                    f"Minority class ratio: {minority_ratio:.4f}"
                )

        return X, y
