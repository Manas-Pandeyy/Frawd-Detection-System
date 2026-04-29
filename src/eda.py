import os

import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df, output_dir: str = "reports") -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1) Class imbalance chart.
    plt.figure(figsize=(7, 4))
    sns.countplot(x="Class", data=df)
    plt.title("Fraud (1) vs Normal (0) Transaction Count")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=160)
    plt.close()

    # 2) Amount distribution by class.
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Class", y="Amount", data=df, showfliers=False)
    plt.title("Transaction Amount by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_by_class.png"), dpi=160)
    plt.close()

    # 3) Time distribution by class.
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Time", hue="Class", bins=60, element="step", stat="density")
    plt.title("Time Distribution by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_distribution.png"), dpi=160)
    plt.close()

    # 4) Correlation heatmap for a subset of columns for readability.
    selected = [c for c in ["Time", "Amount", "V1", "V2", "V3", "V4", "V10", "V12", "V14", "V17", "Class"] if c in df.columns]
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[selected].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=160)
    plt.close()

