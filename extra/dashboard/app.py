from contextlib import ExitStack
from dataclasses import dataclass

import gradio as gr
import pandas as pd


@dataclass
class PlotConfig:
    x_title: str
    y_title: str
    title: str


css = '''
.summary span {
    font-size: 10px;
    padding-top:0;
    padding-bottom:0;
}
'''

summary_desc = '''
## Summary
This table shows the average of the metrics for each model and QPS rate.

Benchmark are run with:
- Prompts: 200±10 tokens length (normal distribution)
- Generation: 200±10 tokens length (normal distribution)
- 120s duration 

Each benchmark is run using a constant arrival rate of requests per second (QPS), 
independently of the number of requests that are being processed (open loop).
'''

df = pd.DataFrame()
summary = pd.DataFrame()
line_plots = []


def plot(model, device) -> pd.DataFrame:
    d = df[(df['model'] == model) & (df['device'] == device)]
    return d


def update_app(device, model):
    res = []
    for plot in line_plots:
        res.append(df[(df['model'] == model) & (df['device'] == device)])
    return res + [summary_table(device)]


def summary_table(device) -> pd.DataFrame:
    rates = [4., 8., 16.]
    data = df[(df['device'] == device) & (df['rate'].isin(rates))]
    data = data.groupby(['model', 'rate']).agg(
        {'inter_token_latency_ms_p90': 'mean', 'time_to_first_token_ms_p90': 'mean',
         'e2e_latency_ms_p90': 'mean', 'token_throughput_secs': 'mean',
         'successful_requests': 'mean', 'error_rate': 'mean'}).reset_index()
    data = data[['model', 'rate', 'inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
                 'token_throughput_secs']]
    data = data.rename(
        columns={'inter_token_latency_ms_p90': 'ITL P90 (ms)', 'time_to_first_token_ms_p90': 'TTFT P90 (ms)',
                 'e2e_latency_ms_p90': 'E2E P90 (ms)', 'token_throughput_secs': 'Throughput (tokens/s)',
                 'successful_requests': 'Successful requests', 'error_rate': 'Error rate (%)', 'model': 'Model',
                 'rate': 'QPS'})
    return data


def load_data() -> pd.DataFrame:
    data = pd.read_parquet('results.parquet')
    # remove warmup and throughput
    data = data[(data['id'] != 'warmup') & (data['id'] != 'throughput')]
    # only keep constant rate
    data = data[data['executor_type'] == 'ConstantArrivalRate']
    return data


if __name__ == '__main__':
    metrics = {
        "inter_token_latency_ms_p90": PlotConfig(title="Inter Token Latency P90 (lower is better)", x_title="QPS",
                                                 y_title="Time (ms)"),
        "time_to_first_token_ms_p90": PlotConfig(title="TTFT P90 (lower is better)", x_title="QPS",
                                                 y_title="Time (ms)"),
        "e2e_latency_ms_p90": PlotConfig(title="End to End Latency P90 (lower is better)", x_title="QPS",
                                         y_title="Time (ms)"),
        "token_throughput_secs": PlotConfig(title="Request Output Throughput P90 (higher is better)", x_title="QPS",
                                            y_title="Tokens/s"),
        "successful_requests": PlotConfig(title="Successful requests (higher is better)", x_title="QPS",
                                          y_title="Count"),
        "error_rate": PlotConfig(title="Error rate (lower is better)", x_title="QPS", y_title="%")
    }
    default_df = pd.DataFrame.from_dict(
        {"rate": [1, 2], "inter_token_latency_ms_p90": [10, 20], "engine": ["tgi", "vllm"]})
    df = load_data()
    models = df["model"].unique()
    devices = df["device"].unique()
    with gr.Blocks(css=css, title="TGI benchmarks") as demo:
        with gr.Row():
            header = gr.Markdown("# TGI benchmarks\nBenchmark results for Hugging Face TGI 🤗")
        with gr.Row():
            device = gr.Radio(devices, label="Select device", value="H100")
        with gr.Row():
            summary_desc = gr.Markdown(summary_desc)
        with gr.Row():
            table = gr.DataFrame(
                pd.DataFrame(),
                elem_classes=["summary"],
            )
        with gr.Row():
            details_desc = gr.Markdown("## Details")
        with gr.Row():
            model = gr.Dropdown(list(models), label="Select model", value=models[0])
        i = 0
        with ExitStack() as stack:
            for k, v in metrics.items():
                if i % 2 == 0:
                    stack.close()
                    gs = stack.enter_context(gr.Row())
                line_plots.append(
                    {"component": gr.LinePlot(default_df, label=f'{v.title}', x="rate", y=k,
                                              color="engine", y_title=v.y_title,
                                              color_map={'vLLM': '#2F5BA1', 'TGI': '#FF9D00'}), "model": model.value,
                     "device": device})
                i += 1

        device.change(update_app, [device, model], [item["component"] for item in line_plots] + [table])
        model.change(update_app, [device, model], [item["component"] for item in line_plots] + [table])
        demo.load(update_app, [device, model], [item["component"] for item in line_plots] + [table])

    demo.launch()
