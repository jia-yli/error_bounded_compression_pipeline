import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

cfg_lst = ['cpu', 'gpu']
label_lst = ['CPU-GPU Feedback Loop', 'Full-GPU Feedback Loop']

def plot(results_path, cfg_lst, col_name):
  variable_values = {}
  for cfg in cfg_lst:
    df = pd.read_csv(os.path.join(results_path, f"compression_results_7d_{cfg}_batch_16.csv"))
    for row_idx in range(len(df)):
      variable = df['variable'].values[row_idx]
      value = df[col_name].values[row_idx]
      variable_values.setdefault(variable, []).append(value)

  # --- Plotting ---
  variables = list(variable_values.keys())
  values = np.array(list(variable_values.values()))  # shape: (num_vars, num_batches)

  x = np.arange(len(variables))  # positions for variables
  width = 0.8 / len(cfg_lst)   # width of each bar, leave some space

  plt.figure(figsize=(12, 6))
  for i, cfg in enumerate(cfg_lst):
    plt.bar(
      x + i * width,
      values[:, i],
      width=width,
      label=label_lst[i],
    )

  # Add labels on top of bars
  for i, cfg in enumerate(cfg_lst):
    for j, val in enumerate(values[:, i]):
      if not np.isnan(val):
        plt.text(x[j] + i * width, val, f"{val:.2f}", 
          ha='center', va='bottom', fontsize=9, rotation=90)

  plt.xlabel("Variable")
  plt.ylabel(col_name)
  plt.title(f"{col_name} for Bound = 1x Ensemble Spread")
  plt.xticks(x + width * (len(cfg_lst) - 1) / 2, variables, rotation=90)
  plt.legend()
  plt.grid(axis='y', linestyle='--', alpha=0.6)

  output_path = f"./plots/{col_name}.png"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()

if __name__ == '__main__':
  plot('.', cfg_lst, 'compression_ratio')
  plot('.', cfg_lst, 'compression_bandwidth')
  plot('.', cfg_lst, 'decompression_bandwidth')