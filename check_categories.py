import pandas as pd
import os

print("Files inside Data folder:")
print(os.listdir("Data"))

# Replace the file name below after seeing list above
df = pd.read_csv("Data/sample.csv")

print("\nEmploymentStatus values:")
print(df["EmploymentStatus"].unique())

print("\nMaritalStatus values:")
print(df["MaritalStatus"].unique())