import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    "axes.titleweight": "bold",
    "axes.titlepad": 16,
    "axes.labelpad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.edgecolor": "#e2e8f0",
    "axes.linewidth": 1,
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "text.color": "#1e293b",
    "axes.labelcolor": "#475569",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
})

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="Blues", 
        fmt=".2f", 
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8}, 
        annot_kws={"fontsize": 11, "weight": "medium"},
        ax=ax
    )
    
    ax.set_title("Matriks Korelasi Variabel", fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    return fig

def plot_distribution(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        df[column], 
        kde=True, 
        color="#2563eb",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        bins=10, 
        ax=ax
    )
    ax.set_title(f"Distribusi Data {column}", fontsize=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frekuensi", fontsize=12)
    fig.tight_layout()
    return fig

def plot_actual_vs_predicted(y_actual, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.scatter(
        y_actual, 
        y_pred, 
        alpha=0.75, 
        color="#2563eb", 
        edgecolors="#1e40af",
        linewidth=0.5,
        s=65
    )
    
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    
    ax.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        "--", 
        color="#94a3b8", 
        linewidth=1.5, 
        label="Garis Prediksi Ideal"
    )
    
    ax.set_title("Perbandingan Nilai Aktual vs Prediksi", fontsize=14)
    ax.set_xlabel("Nilai Aktual", fontsize=12)
    ax.set_ylabel("Nilai Prediksi Model", fontsize=12)
    
    legend = ax.legend(frameon=True, fancybox=True, facecolor="white", edgecolor="#e2e8f0")
    legend.get_frame().set_alpha(0.9)
    
    fig.tight_layout()
    return fig