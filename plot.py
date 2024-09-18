import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use('science')
pd.options.mode.copy_on_write = True


def plot(data_files: dict[str, str]):
    df = pd.DataFrame()
    # Load the results
    for key, filename in data_files.items():
        with open(filename, 'r') as f:
            data = json.load(f)
            for result in data['results']:
                entry = result
                [config] = pd.json_normalize(result['config']).to_dict(orient='records')
                entry.update(config)
                entry['engine'] = key
                del entry['config']
                df = pd.concat([df, pd.DataFrame(entry, index=[0])])

    # Filter the results
    constant_rate = df[
        (df['executor_type'] == 'ConstantArrivalRate') & (df['id'] != 'warmup') & (df['id'] != 'throughput')]
    constant_vus = df[(df['executor_type'] == 'ConstantVUs') & (df['id'] != 'warmup') & (df['id'] != 'throughput')]
    if len(constant_rate) > 0:
        plot_inner('Requests/s', 'rate', constant_rate, 'Constant Rate benchmark')
    if len(constant_vus) > 0:
        plot_inner('VUs', 'max_vus', constant_vus, 'Constant VUs benchmark')


def plot_inner(x_title, x_key, results, chart_title):
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.tight_layout(pad=6.0)
    fig.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.05, top=0.92)
    # compute error rate
    results['error_rate'] = results['failed_requests'] / (
            results['failed_requests'] + results['successful_requests']) * 100.0

    metrics = ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
               'token_throughput_secs',
               'successful_requests', 'error_rate']

    titles = ['Inter Token Latency P90 (lower is better)', 'TTFT P90 (lower is better)',
              'End to End Latency P90 (lower is better)',
              'Token Throughput (higher is better)', 'Successful requests', 'Error Rate % (lower is better)']

    labels = ['Time (ms)', 'Time (ms)', 'Time (ms)', 'Tokens/s', 'Count', '%']

    colors = ['#2F5BA1']

    # Plot each metric in its respective subplot
    for ax, metric, title, label in zip(axs.flatten(), metrics, titles, labels):
        for i, engine in enumerate(results['engine'].unique()):
            df_sorted = results[results['engine'] == engine].sort_values(by=x_key)
            ax.plot(df_sorted[x_key], df_sorted[metric], marker='o', markersize=2, color=colors[i % len(colors)] if engine!='tgi' else '#FF9D00',
                    label=f"{engine}")
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=0)
        ax.set_ylabel(label)
        ax.set_xlabel(x_title)
        ax.set_ylim(0)
        # rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-90)

        # Enhance y-axis: more ticks for better granularity
        y_vals = ax.get_yticks()
        ax.set_yticks(
            np.linspace(y_vals[0], y_vals[-1], 10))  # Set 10 ticks spread between the min and max of current y-ticks

        # Add grid lines for better readability
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)  # Ensure grid lines are below the bars
        ax.legend(title='Engine', loc='upper right')
    plt.suptitle(chart_title, fontsize=16)

    plt.show()


if __name__ == '__main__':
    # list json files in results directory
    data_files = {}
    for filename in os.listdir('results'):
        if filename.endswith('.json'):
            data_files[filename.split('.')[0]] = f'results/{filename}'
    plot(data_files)
