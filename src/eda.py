
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "C:/Users/R O G/Desktop/2nd Year/MachineLearning/CW/cm2604-telco-churn/data/telco-customer-churn.csv"
FIG_DIR = "C:/Users/R O G/Desktop/2nd Year/MachineLearning/CW/cm2604-telco-churn/outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

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

    #figure 01 target class distribution
    plt.figure(figsize=[7,5])
    ax=sns.countplot(data=df, x="Churn", hue="Churn", palette=["#2b5c8f", "#d95f02"], legend=False)

    plt.title("Customer Churn Class Distribution")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_churn_distribution.png", dpi=300)
    plt.close()

    #figure 02 churn by contract type and internet service
    fig,axes=plt.subplots(1,2,figsize=(12,4))

    sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0], palette=["#2b5c8f", "#d95f02"])
    axes[0].set_title("Churn by Contract Type")
    axes[0].tick_params(axis="x", rotation=15)

    sns.countplot(data=df,x="InternetService", hue="Churn",ax=axes[1], palette=["#2b5c8f", "#d95f02"])
    axes[1].set_title("Churn by Internet Service")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_contract_and_internetservice_churn.png", dpi=300)
    plt.close()

    #figure 03 tenure and monthly charges by churn
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    sns.boxplot(data=df,y="tenure",x="Churn",ax=axes[0], palette=["#2b5c8f", "#d95f02"],hue="Churn",legend=False)
    axes[0].set_title("Tenure vs Churn")

    sns.boxplot(data=df,y="MonthlyCharges",x="Churn",ax=axes[1],palette=["#2b5c8f", "#d95f02"],hue="Churn",legend=False)
    axes[1].set_title("Monthly Charges vs Churn")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_tenure_and_monthlycharges_churn.png", dpi=300)
    plt.close()

    #figure 04 correlation heatmap
    df_corr = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num"]].copy()
    df_corr["Churn"] = (df["Churn"] == "Yes").astype(int)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1
    )
    plt.title("Correlation Heatmap of Numerical Features & Churn")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_correlation_heatmap.png", dpi=300)
    plt.close()

    return df

if __name__ == "__main__":
    run_eda(DATA_PATH)