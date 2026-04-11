from typing import Optional, Union, List

import pandas as pd
import yaml
from pydantic import BaseModel
from collections import defaultdict

import json

from pogema_toolbox.registry import ToolboxRegistry


class View(BaseModel):
    drop_keys: list = ['seed']
    round_digits: int = 2
    rename_fields: dict = {"algorithm": "Algorithm",
                           "num_agents": "Number of Agents",
                           "avg_throughput": "Average Throughput",
                           "runtime": "Runtime (seconds)",
                           "max_episode_steps": "Episode Length"
                           }
    rename_algorithms: dict = {}
    sort_by: Optional[Union[str, List[str]]] = None


def drop_na(df):
    # Drop columns with any NaN values
    cols_before_drop_na = set(df.columns)
    df = df.dropna(axis=1, how='any')
    cols_after_drop_na = set(df.columns)

    # Log dropped columns
    dropped_cols = cols_before_drop_na - cols_after_drop_na
    for col in dropped_cols:
        ToolboxRegistry.warning(f"Column '{col}' dropped due to missing values")

    return df


def eval_logs_to_pandas(eval_configs):
    data = {}
    for idx, config in enumerate(eval_configs):
        data[idx] = {**config['env_grid_search'], 'algorithm': config['algorithm']}

        # Adding metrics separately to skip possible lists of metrics (e.g. every step throughput)
        for key, value in config['metrics'].items():
            if isinstance(value, list):
                continue
            data[idx][key] = value
    return pd.DataFrame.from_dict(data, orient='index')


def load_from_folder(folder_path):
    eval_config_path = folder_path / (folder_path.name + '.yaml')
    # check if the file with name current_dir_name.name + '.yaml' exists
    assert eval_config_path.exists(), f'Config file {eval_config_path} does not exist'

    with open(eval_config_path, 'r') as f:
        evaluation_config = yaml.safe_load(f)

    results = []
    # Load results from *.json files
    for file in folder_path.glob('*.json'):
        with open(file, 'r') as f:
            results += json.load(f)
    return results, evaluation_config


def check_seeds(results):
    """
    Analyze seed consistency across results.

    Returns:
        dict with keys:
            - problem_count: int
            - seed_data: dict mapping (map_name, num_agents, seed) -> {algo: [results]}
            - algorithms: sorted list of algorithm names
            - errors: list of error strings (if data is malformed)
    """
    seed_data = defaultdict(lambda: defaultdict(list))
    algorithms = set()
    errors = []

    for res in results:
        if 'env_grid_search' not in res:
            errors.append("env_grid_search data missing")
            continue

        env_data = res.get('env_grid_search', {})

        if 'seed' not in env_data:
            errors.append("No seed in env_grid_search data")
            continue

        map_name = env_data.get('map_name', None)
        num_agents = env_data.get('num_agents', None)
        seed = env_data['seed']
        algo = res['algorithm']
        algorithms.add(algo)
        seed_data[(map_name, num_agents, seed)][algo].append(res)

    problem_count = 0
    sorted_algorithms = sorted(algorithms)
    for (_map_name, _num_agents, _seed), algos in seed_data.items():
        for algo in sorted_algorithms:
            if algo not in algos:
                problem_count += 1
            elif len(algos[algo]) > 1:
                problem_count += 1

    return {
        'problem_count': problem_count,
        'seed_data': dict(seed_data),
        'algorithms': sorted_algorithms,
        'errors': errors,
    }
