import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')


def plot():
    # Load the results
    with open('results/results.json', 'r') as f:
        data = json.load(f)

    results_filtered = [result for result in data['results'] if
                        result['id'] != 'warmup' and result['id'] != 'throughput']
    constant_rate = [result for result in results_filtered if result['executor_type'] == 'ConstantArrivalRate']
    constant_rate_x = [result['config']['rate'] for result in constant_rate]
    constant_vus = [result for result in results_filtered if result['executor_type'] == 'ConstantVUs']
    constant_vus_x = [result['config']['vus'] for result in constant_vus]
    if len(constant_rate) > 0:
        plot_inner('Requests/s', constant_rate_x, constant_rate, 'Constant Rate benchmark')
    if len(constant_vus) > 0:
        plot_inner('VUs', constant_vus_x, constant_vus, 'Constant VUs benchmark')


def plot_inner(x_name, x_values, results, chart_title):
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.tight_layout(pad=6.0)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.15, top=0.92)
    # compute error rate
    for result in results:
        result['error_rate'] = result['failed_requests'] / (
                result['failed_requests'] + result['successful_requests']) * 100.0

    metrics = ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90', 'token_throughput_secs',
               'successful_requests', 'error_rate']

    titles = ['Inter Token Latency P90 (lower is better)', 'TTFT P90 (lower is better)', 'End to End Latency P90 (lower is better)',
              'Token Throughput (higher is better)', 'Successful requests', 'Error Rate % (lower is better)']

    labels = ['Time (ms)', 'Time (ms)', 'Time (ms)', 'Tokens/s', 'Count', '%']

    colors = ['#FF9D00', '#2F5BA1']

    # Plot each metric in its respective subplot
    for ax, metric, title, label in zip(axs.flatten(), metrics, titles, labels):
        data = list(map(lambda result: result[metric], results))
        ax.plot(x_values, data, marker='o', color=colors[0])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=0)
        ax.set_ylabel(label)
        ax.set_xlabel(x_name)
        # rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-90)

        # Enhance y-axis: more ticks for better granularity
        y_vals = ax.get_yticks()
        ax.set_yticks(
            np.linspace(y_vals[0], y_vals[-1], 10))  # Set 10 ticks spread between the min and max of current y-ticks

        # Add grid lines for better readability
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)  # Ensure grid lines are below the bars
    plt.suptitle(chart_title, fontsize=16)

    plt.show()


if __name__ == '__main__':
    plot()
