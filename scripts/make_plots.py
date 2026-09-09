# scripts/make_plots.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_probe_results(probe_results_path: Path, output_path: str):
    if not Path(probe_results_path).exists():
        raise ValueError(f'{probe_results_path} does not exist. Failed to plot!')

    if not Path(output_path).exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    probe_results = pd.read_csv(probe_results_path)

    metric_cols = [c for c in probe_results.columns if c != 'layer']
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metric_cols:
        ax.plot(probe_results['layer'], probe_results[metric], marker='o', label=metric)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('Probe performance by layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path(output_path).with_suffix('.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    probe_plot_path = 'plots/probe_plot.png'
    probe_results_path = 'data/layer_auroc.csv'
    probe_baseline_comparison_path = 'data/baseline_comparison.csv'

    plot_probe_results(probe_results_path, probe_plot_path)