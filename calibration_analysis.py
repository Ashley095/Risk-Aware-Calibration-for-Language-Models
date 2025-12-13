"""
Calibration & Abstention Metrics Analysis
- Metrics: ECE, NLL, Brier, AURC
- Each metric gets its own combined plot (TS vs NoT for all models)
- Saves all results in `analysis_results`
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

# =========================================
# CONFIGURATION
# =========================================

# Ensure paths are resolved relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "all_model_runs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_N_LOAD = 1000

MODEL_TAGS = ['deberta_nli', 'roberta_race', 'flan_t5_xl', 'phi_3.5_mini']

MODEL_STYLE = {
    'deberta_nli': {'name': 'DeBERTa-NLI', 'marker': 'o', 'color_ts': 'darkblue', 'color_nots': 'lightblue'},
    'roberta_race': {'name': 'RoBERTa-Race', 'marker': 's', 'color_ts': 'purple', 'color_nots': 'plum'},
    'flan_t5_xl': {'name': 'T5-XL', 'marker': '^', 'color_ts': 'firebrick', 'color_nots': 'lightcoral'},
    'phi_3.5_mini': {'name': 'Phi-3.5-Mini', 'marker': 'D', 'color_ts': 'teal', 'color_nots': 'turquoise'}
}

# =========================================
# DATA LOADING
# =========================================

def load_model_summary(model_tag):
    """Load all metrics (ECE, NLL, Brier, AURC) for a model across calibration ratios"""
    model_dir = os.path.join(RESULTS_DIR, model_tag)
    if not os.path.isdir(model_dir):
        print(f"⚠ Missing directory: {model_dir}")
        return pd.DataFrame()

    rows = []
    ratio_folders = sorted(glob.glob(os.path.join(model_dir, "cal_ratio_*")))

    for folder in ratio_folders:
        ratio = int(folder.split("_")[-1])
        cal_samples = int(MAX_N_LOAD * ratio / 100)

        # --- Calibration metrics ---
        calib_csv = os.path.join(folder, f"summary_calibration_{model_tag}.csv")
        if not os.path.exists(calib_csv):
            continue
        df_calib = pd.read_csv(calib_csv)
        ts_calib = df_calib[df_calib["Method"] == "TS"].iloc[0]
        nots_calib = df_calib[df_calib["Method"] == "NoTS"].iloc[0]

        # --- Abstention metrics (AURC) ---
        abstain_csv = os.path.join(folder, f"summary_abstain_{model_tag}.csv")
        if os.path.exists(abstain_csv):
            df_abst = pd.read_csv(abstain_csv)
            ts_aurc = df_abst[df_abst["Method"]=="TS"].iloc[0]["AURC"]
            not_aurc = df_abst[df_abst["Method"]=="NoT"].iloc[0]["AURC"]
        else:
            ts_aurc = None
            not_aurc = None

        rows.append({
            "Ratio (%)": ratio,
            "CAL Samples (N)": cal_samples,
            "ECE_TS": ts_calib["ECE"],
            "ECE_NoTS": nots_calib["ECE"],
            "NLL_TS": ts_calib["NLL"],
            "NLL_NoTS": nots_calib["NLL"],
            "Brier_TS": ts_calib.get("Brier", None),
            "Brier_NoTS": nots_calib.get("Brier", None),
            "AURC_TS": ts_aurc,
            "AURC_NoT": not_aurc
        })

    df = pd.DataFrame(rows).sort_values(by="Ratio (%)").reset_index(drop=True)
    return df

# Load summaries for all models
summaries = {tag: load_model_summary(tag) for tag in MODEL_TAGS}

# =========================================
# PLOTTING FUNCTION
# =========================================

def plot_metric(metric_ts, metric_nots, ylabel, title, filename):
    plt.figure(figsize=(8,6))
    for tag in MODEL_TAGS:
        df = summaries[tag]
        if df.empty:
            continue
        style = MODEL_STYLE[tag]
        x = df["CAL Samples (N)"]
        plt.plot(x, df[metric_nots], f"{style['marker']}--", color=style['color_nots'], label=f"{style['name']} (NoTS/NoT)", alpha=0.7)
        plt.plot(x, df[metric_ts], f"{style['marker']}-", color=style['color_ts'], label=f"{style['name']} (TS)", linewidth=2.2)

    plt.title(title)
    plt.xlabel("Calibration Samples (N)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160)
    plt.show()
    print(f"✓ Saved: {filename}")

# =========================================
# PLOT ALL METRICS
# =========================================

plot_metric("ECE_TS", "ECE_NoTS", "ECE", "Expected Calibration Error (ECE)", "ece_comparison.png")
plot_metric("NLL_TS", "NLL_NoTS", "Negative Log-Likelihood (NLL)", "NLL Comparison", "nll_comparison.png")
plot_metric("Brier_TS", "Brier_NoTS", "Brier Score", "Brier Score Comparison", "brier_comparison.png")
plot_metric("AURC_TS", "AURC_NoT", "AURC", "AURC Comparison", "aurc_comparison.png")

# =========================================
# SUMMARY TABLE OF BEST CONFIGS (Lowest ECE_TS)
# =========================================

best_configs = []
for tag in MODEL_TAGS:
    df = summaries[tag]
    if df.empty:
        continue
    best_idx = df["ECE_TS"].idxmin()
    best_row = df.loc[best_idx]
    best_configs.append({
        "Model": MODEL_STYLE[tag]["name"],
        "Best CAL Ratio (%)": best_row["Ratio (%)"],
        "Best CAL Samples": best_row["CAL Samples (N)"],
        "ECE_TS": best_row["ECE_TS"],
        "NLL_TS": best_row["NLL_TS"],
        "Brier_TS": best_row["Brier_TS"],
        "AURC_TS": best_row["AURC_TS"]
    })

best_df = pd.DataFrame(best_configs)
best_df.to_csv(os.path.join(OUTPUT_DIR, "best_configurations.csv"), index=False)
print("\nBest configurations saved to 'best_configurations.csv':\n")
print(best_df)

# =========================================
# PLOT ALL METRICS IN A 2x2 GRID
# =========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metric_info = [
    ("ECE_TS", "ECE_NoTS", "ECE", "Expected Calibration Error (ECE)"),
    ("NLL_TS", "NLL_NoTS", "NLL", "Negative Log-Likelihood"),
    ("Brier_TS", "Brier_NoTS", "Brier", "Brier Score"),
    ("AURC_TS", "AURC_NoT", "AURC", "AURC")
]

for ax, (metric_ts, metric_nots, ylabel, title) in zip(axes.flatten(), metric_info):
    for tag in MODEL_TAGS:
        df = summaries[tag]
        if df.empty:
            continue
        style = MODEL_STYLE[tag]
        x = df["CAL Samples (N)"]
        ax.plot(x, df[metric_nots], f"{style['marker']}--", color=style['color_nots'],
                label=f"{style['name']} (NoTS/NoT)", alpha=0.7)
        ax.plot(x, df[metric_ts], f"{style['marker']}-", color=style['color_ts'],
                label=f"{style['name']} (TS)", linewidth=2.2)
    ax.set_title(title)
    ax.set_xlabel("Calibration Samples (N)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

fig.suptitle("Calibration & Abstention Metrics Across Models (TS vs NoT)", fontsize=16, y=1, fontweight='bold')

# Combine legends from all subplots and place directly below the title
handles, labels = [], []
for ax in axes.flatten():
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.97))

plt.tight_layout(rect=[0,0,1,0.95])
grid_filename = os.path.join(OUTPUT_DIR, "metrics_2x2_grid.png")
plt.savefig(grid_filename, dpi=160)
plt.show()
print(f"Saved combined 2x2 grid: {grid_filename}")

# =========================================
# CREATE SUMMARY CSV OF TS IMPROVEMENTS (WITH AURC)
# =========================================

summary_rows = []

for tag in MODEL_TAGS:
    df = summaries[tag]
    if df.empty:
        continue
    style = MODEL_STYLE[tag]
    
    # Take the average over all calibration ratios
    avg_no_ts = df[["NLL_NoTS", "ECE_NoTS", "Brier_NoTS", "AURC_NoT"]].mean()
    avg_ts    = df[["NLL_TS", "ECE_TS", "Brier_TS", "AURC_TS"]].mean()

    summary_rows.append({
        "Model": style["name"],
        "NLL_NoTS": avg_no_ts["NLL_NoTS"],
        "NLL_TS": avg_ts["NLL_TS"],
        "ECE_NoTS": avg_no_ts["ECE_NoTS"],
        "ECE_TS": avg_ts["ECE_TS"],
        "Brier_NoTS": avg_no_ts["Brier_NoTS"],
        "Brier_TS": avg_ts["Brier_TS"],
        "AURC_NoT": avg_no_ts["AURC_NoT"],
        "AURC_TS": avg_ts["AURC_TS"],
    })

summary_df = pd.DataFrame(summary_rows)

# Compute average percentage improvement: (NoTS - TS) / NoTS * 100
summary_df["ΔNLL (%)"]   = 100 * (summary_df["NLL_NoTS"] - summary_df["NLL_TS"]) / summary_df["NLL_NoTS"]
summary_df["ΔECE (%)"]   = 100 * (summary_df["ECE_NoTS"] - summary_df["ECE_TS"]) / summary_df["ECE_NoTS"]
summary_df["ΔBrier (%)"] = 100 * (summary_df["Brier_NoTS"] - summary_df["Brier_TS"]) / summary_df["Brier_NoTS"]
summary_df["ΔAURC (%)"]  = 100 * (summary_df["AURC_NoT"] - summary_df["AURC_TS"]) / summary_df["AURC_NoT"]

# Reorder columns for CSV
summary_df = summary_df[[
    "Model",
    "NLL_NoTS", "NLL_TS", "ΔNLL (%)",
    "ECE_NoTS", "ECE_TS", "ΔECE (%)",
    "Brier_NoTS", "Brier_TS", "ΔBrier (%)",
    "AURC_NoT", "AURC_TS", "ΔAURC (%)"
]]

summary_csv_path = os.path.join(OUTPUT_DIR, "ts_improvement_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, float_format="%.3f")
print(f"Saved TS improvement summary (with AURC): {summary_csv_path}")
print(summary_df)
