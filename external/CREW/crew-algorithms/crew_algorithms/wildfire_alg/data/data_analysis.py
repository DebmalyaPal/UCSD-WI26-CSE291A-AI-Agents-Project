#!/usr/bin/env python3
"""
Generate per-level comparison plots and export summary statistics as CSVs from the latest action_reward.csv of each seed.

Assumes this script lives alongside a 'Logs/' directory structured as:
Logs/
  algorithm_name/
    level_name/
      seed/
        timestamp/
          action_reward.csv

Outputs go into './plots/'.
  - Line plots:                   plots/{level}_{metric}.png
  - Final-value bars:             plots/{level}_final_{metric}_bar.png
  - Rate bars (avg per timestep): plots/{level}_rate_{metric}_bar.png
  - Run-length bars:              plots/{level}_run_length_bar.png
  - Rate vs Agents plots:         plots/{metric}_vs_agents.png & plots/rate_vs_agents.csv
  - CSV summary:                  plots/final_score_stats.csv
                                   plots/rate_stats.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -- Configuration ------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR   = os.path.join(BASE_DIR, "Logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

# Map from level name to number of agents
AGENT_COUNT_MAP = {
    "Level_1_small": 3,
    "Level_1_large": 10,
    "Level_2_small": 3,
    "Level_2_large": 7,
    "Level_3_small": 3,
    "Level_3_large": 5,
    "Level_4_small": 7,
    "Level_4_large": 14,
    "Level_5_small": 3,
    "Level_5_large": 5,
    "Level_6": 6,
    "Level_7": 8,
    "Level_8": 7,
    "Level_9": 8,
    "Level_10": 14,
    "Level_11": 14,
    "Level_12": 15
}
# Levels to exclude from rate-vs-agents
# This is because the number of agents may not be constant across trajectories
EXCLUDED_LEVELS = {"Level_6", "Level_7", "Level_8", "Level_9", "Level_10","Level_11", "Level_12"}

METRICS = [
    "cumulative_score",
    "cumulative_api_calls",
    "cumulative_input_tokens",
    "cumulative_output_tokens"
]
RATE_METRICS = [
    "cumulative_api_calls",
    "cumulative_input_tokens",
    "cumulative_output_tokens"
]

# -- Data Collection ----------------------------------------------------------

def get_latest_timestamp(seed_path):
    ts = [d for d in os.listdir(seed_path)
          if os.path.isdir(os.path.join(seed_path, d))]
    return max(ts) if ts else None


def collect_data():
    data = {}
    for algo in os.listdir(LOGS_DIR):
        algo_path = os.path.join(LOGS_DIR, algo)
        if not os.path.isdir(algo_path):
            continue
        for level in os.listdir(algo_path):
            level_path = os.path.join(algo_path, level)
            if not os.path.isdir(level_path):
                continue
            for seed in os.listdir(level_path):
                seed_path = os.path.join(level_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                ts = get_latest_timestamp(seed_path)
                if not ts:
                    continue
                csv_fp = os.path.join(seed_path, ts, "action_reward.csv")
                if not os.path.isfile(csv_fp):
                    continue
                df = pd.read_csv(csv_fp)
                data.setdefault(level, {}).setdefault(algo, []).append(df)
    return data

# -- Plotting Functions -------------------------------------------------------

def plot_line_metrics(data):
    for level, algos in data.items():
        max_len = max(len(df) for dfs in algos.values() for df in dfs)
        for metric in METRICS:
            plt.figure()
            for algo, dfs in algos.items():
                arrs = []
                for df in dfs:
                    vals = df[metric].values
                    if len(vals) < max_len:
                        pad = np.full(max_len - len(vals), vals[-1])
                        vals = np.concatenate([vals, pad])
                    else:
                        vals = vals[:max_len]
                    arrs.append(vals)
                arr = np.vstack(arrs)
                mean = arr.mean(axis=0)
                vmin = arr.min(axis=0)
                vmax = arr.max(axis=0)
                x = np.arange(max_len)
                plt.plot(x, mean, label=algo)
                plt.fill_between(x, vmin, vmax, alpha=0.2)
            plt.title(f"{level} — {metric.replace('_',' ').title()}")
            plt.xlabel("Timestep")
            plt.ylabel(metric.replace('_',' ').title())
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_{metric}.png"))
            plt.close()


def plot_bar_metrics(data):
    for level, algos in data.items():
        for metric in METRICS:
            means, lowers, uppers, labels = [], [], [], []
            for algo, dfs in algos.items():
                finals = [df[metric].iloc[-1] for df in dfs]
                m, mn, mx = np.mean(finals), np.min(finals), np.max(finals)
                means.append(m)
                lowers.append(max(0, m - mn))
                uppers.append(max(0, mx - m))
                labels.append(algo)
            x = np.arange(len(labels))
            plt.figure()
            plt.bar(x, means, yerr=[lowers, uppers], capsize=5)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.title(f"{level} — Final Cumulative Score")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_final_score_bar.png"))
            plt.close()


def plot_run_length_bars(data):
    for level, algos in data.items():
        means, lowers, uppers, labels = [], [], [], []
        for algo, dfs in algos.items():
            lengths = [len(df) for df in dfs]
            m, mn, mx = np.mean(lengths), np.min(lengths), np.max(lengths)
            means.append(m)
            lowers.append(max(0, m - mn))
            uppers.append(max(0, mx - m))
            labels.append(algo)
        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, means, yerr=[lowers, uppers], capsize=5)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.title(f"{level} — Run Length (Timesteps)")
        plt.ylabel("# Timesteps")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_run_length_bar.png"))
        plt.close()


def plot_rate_bar_metrics(data):
    for level, algos in data.items():
        for metric in RATE_METRICS:
            means, lowers, uppers, labels = [], [], [], []
            for algo, dfs in algos.items():
                rates = [df[metric].iloc[-1] / len(df) for df in dfs]
                m, mn, mx = np.mean(rates), np.min(rates), np.max(rates)
                means.append(m)
                lowers.append(max(0, m - mn))
                uppers.append(max(0, mx - m))
                labels.append(algo)
            x = np.arange(len(labels))
            plt.figure()
            plt.bar(x, means, yerr=[lowers, uppers], capsize=5)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.title(f"{level} — Avg per Timestep")
            plt.ylabel("Value per Timestep")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_rate_{metric}_bar.png"))
            plt.close()


def plot_rate_vs_agents(data):
    """Plot and export average rate metrics vs number of agents for each algorithm as a dot-line plot and CSV."""
    # prepare CSV data structure
    agents = sorted({cnt for lvl,cnt in AGENT_COUNT_MAP.items() if lvl not in EXCLUDED_LEVELS})
    algos = sorted({alg for lvl in data for alg in data[lvl]})
    csv_df = pd.DataFrame(index=agents, columns=algos)

    for metric in RATE_METRICS:
        plt.figure()
        for algo in algos:
            rates_by_agents = {}
            for level, algos_data in data.items():
                if level in EXCLUDED_LEVELS or algo not in algos_data:
                    continue
                count = AGENT_COUNT_MAP.get(level)
                for df in algos_data[algo]:
                    rate = df[metric].iloc[-1] / len(df)
                    rates_by_agents.setdefault(count, []).append(rate)
            means = {c: np.mean(rates_by_agents[c]) for c in rates_by_agents}
            # fill CSV
            for c in means:
                csv_df.at[c, algo] = f"{means[c]:.2f}"
            # plot line
            counts = sorted(means)
            plt.plot(counts, [means[c] for c in counts], marker='o', label=algo)
        plt.title(f"Average {metric.replace('_',' ').title()} per Timestep vs # Agents")
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metric.replace('_',' ').title()} per Timestep")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_vs_agents.png"), dpi=300)
        plt.close()
        # export CSV for this metric
        out_csv = os.path.join(OUTPUT_DIR, f"{metric}_vs_agents.csv")
        csv_df.to_csv(out_csv)
        print(f"Saved {metric}_vs_agents to {out_csv}")



def export_final_score_stats(data):
    levels = sorted(data.keys())
    algos = sorted({algo for lvl in data for algo in data[lvl]})
    df_stats = pd.DataFrame(index=levels, columns=algos)
    for lvl in levels:
        for algo in algos:
            dfs = data[lvl].get(algo, [])
            if not dfs:
                df_stats.at[lvl, algo] = ""
            else:
                finals = [d['cumulative_score'].iloc[-1] for d in dfs]
                m, std = np.mean(finals), np.std(finals, ddof=1)
                df_stats.at[lvl, algo] = f"{m:.2f}±{std:.2f}"
    out_fp = os.path.join(OUTPUT_DIR, 'final_score_stats.csv')
    df_stats.to_csv(out_fp)
    print(f"Saved final score stats to {out_fp}")


def export_rate_stats(data):
    algos = sorted({algo for lvl in data for algo in data[lvl]})
    df_rate = pd.DataFrame(index=algos, columns=RATE_METRICS)
    for algo in algos:
        for met in RATE_METRICS:
            vals = []
            for lvl in data:
                for d in data[lvl].get(algo, []):
                    vals.append(d[met].iloc[-1] / len(d))
            arr = np.array(vals)
            df_rate.at[algo, met] = f"{arr.mean():.2f}±{arr.std():.2f}"
    out_fp = os.path.join(OUTPUT_DIR, 'rate_stats.csv')
    df_rate.to_csv(out_fp)
    print(f"Saved rate stats to {out_fp}")

# -- Main ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = collect_data()
    if not data:
        print(f"No data under {LOGS_DIR}")
        return
    plot_line_metrics(data)
    plot_bar_metrics(data)
    plot_run_length_bars(data)
    plot_rate_bar_metrics(data)
    plot_rate_vs_agents(data)
    export_final_score_stats(data)
    export_rate_stats(data)
    print("All plots and CSV summaries generated in", OUTPUT_DIR)

if __name__ == '__main__':
    main()
