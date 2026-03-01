import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class AutoEDA:
    def __init__(self, output_path="eda_reports"):
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def identify_column_types(self, df):
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        logging.info(f"Numerical columns detected: {numerical_cols}")
        logging.info(f"Categorical columns detected: {categorical_cols}")

        return numerical_cols, categorical_cols

    def plot_distributions(self, df, numerical_cols):
        for col in numerical_cols:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(f"{self.output_path}/{col}_distribution.png")
            plt.close()

        logging.info("Numerical distributions saved.")

    def correlation_heatmap(self, df, numerical_cols):
        if len(numerical_cols) > 1:
            plt.figure(figsize=(8,6))
            corr = df[numerical_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.savefig(f"{self.output_path}/correlation_heatmap.png")
            plt.close()

            logging.info("Correlation heatmap saved.")

    def check_class_imbalance(self, df, target_column):
        if target_column in df.columns:
            counts = df[target_column].value_counts()

            plt.figure()
            sns.countplot(x=df[target_column])
            plt.title("Target Class Distribution")
            plt.savefig(f"{self.output_path}/class_distribution.png")
            plt.close()

            logging.info(f"Class distribution: {counts.to_dict()}")

            return counts

        return None
