
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "C:/Users/R O G/Desktop/2nd Year/MachineLearning/CW/cm2604-telco-churn/data/telco-customer-churn.csv"
FIG_DIR = "C:/Users/R O G/Desktop/2nd Year/MachineLearning/CW/cm2604-telco-churn/outputs/figures"

def run_eda(path):
    df=pd.read_csv(path)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn data types: ")
    print(f"{df.dtypes}")

    print(f"\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


    #details of where total charges is null
    df["TotalCharges_num"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_count=df["TotalCharges_num"].isnull().sum()
    print(f"\nBlank Count of Columns: {blank_count}")
    if blank_count>0:
        print(
            df[df["TotalCharges_num"].isnull()][
                ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
            ]
        )

    #target class distribution
    churn_counts=df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print(f"\nTarget class distribution")
    for label in churn_counts.index:
        print(f"  {label}: {churn_counts[label]} ({churn_pct[label]:.2f}%)")

    #numerical summary
    print("\nNumerical Summary:")
    print(df[["tenure", "MonthlyCharges", "TotalCharges_num"]].describe())


run_eda(DATA_PATH)