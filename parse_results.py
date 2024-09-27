import json
import os
from operator import index

import pandas as pd
from pandas import DataFrame


def build_df(model: str, data_files: dict[str, str]) -> pd.DataFrame:
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
                entry['model'] = model
                del entry['config']
                df = pd.concat([df, pd.DataFrame(entry, index=[0])])
    return df


def build_results_df() -> pd.DataFrame:
    results_dir = 'results'
    df = pd.DataFrame()
    # list directories
    directories = [f'{results_dir}/{d}' for d in os.listdir(results_dir) if os.path.isdir(f'{results_dir}/{d}')]
    for directory in directories:
        # list json files in results directory
        data_files = {}
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                data_files[filename.split('.')[-2]] = f'{directory}/{filename}'
        df = pd.concat([df, build_df(directory.split('/')[-1], data_files)])
    return df


if __name__ == '__main__':
    df = build_results_df()
    df['device'] = df['model'].apply(lambda x: 'H100')
    df['error_rate'] = df['failed_requests'] / (df['failed_requests'] + df['successful_requests']) * 100.0
    df.to_parquet('results.parquet')
