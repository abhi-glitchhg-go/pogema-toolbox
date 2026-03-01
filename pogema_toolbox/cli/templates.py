import yaml


class _CompactListDumper(yaml.SafeDumper):
    """YAML dumper that renders lists in flow style ([1, 2, 3]) and dicts in block style."""
    pass


_CompactListDumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True),
)


def render_eval_config(config: dict) -> str:
    return yaml.dump(config, Dumper=_CompactListDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _make_import_line(import_path: str) -> tuple[str, str]:
    """Convert 'module:ClassName' to ('from module import ClassName', 'ClassName')."""
    module, cls = import_path.rsplit(":", 1)
    return f"from {module} import {cls}", cls


def render_run_script(
    algo_name: str,
    algo_key: str,
    algo_mode: str,
    algo_import_path: str | None = None,
    algo_config_import_path: str | None = None,
) -> str:
    import_lines = []
    register_args = ""

    if algo_mode == "template":
        import_lines.append("from algorithm import MyAlgorithm")
        register_args = f'"{algo_name}", MyAlgorithm'
    else:
        line, cls = _make_import_line(algo_import_path)
        import_lines.append(line)
        register_args = f'"{algo_name}", {cls}'

        if algo_config_import_path:
            cfg_line, cfg_cls = _make_import_line(algo_config_import_path)
            import_lines.append(cfg_line)
            register_args += f", {cfg_cls}"

    imports_block = "\n".join(import_lines)

    return f'''"""Run evaluation for {algo_name}."""
from pathlib import Path

import yaml

from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.registry import ToolboxRegistry

{imports_block}


def main():
    ToolboxRegistry.setup_logger(level="INFO")

    # Register environment
    ToolboxRegistry.register_env("Pogema-v0", create_env_base, Environment)

    # Register algorithm
    ToolboxRegistry.register_algorithm({register_args})

    # Load config and run
    config_path = Path(__file__).parent / "eval_config.yaml"
    eval_dir = Path(__file__).parent / "results"

    with open(config_path) as f:
        evaluation_config = yaml.safe_load(f)

    evaluation(evaluation_config, eval_dir=eval_dir)


if __name__ == "__main__":
    main()
'''


def render_render_script() -> str:
    return '''"""Re-render result views from saved JSON files."""
import json
from pathlib import Path

import yaml

from pogema_toolbox.evaluator import run_views


def main():
    eval_dir = Path(__file__).parent / "results"
    config_path = Path(__file__).parent / "eval_config.yaml"

    with open(config_path) as f:
        evaluation_config = yaml.safe_load(f)

    # Load all JSON result files
    results = []
    for json_file in sorted(eval_dir.glob("*.json")):
        with open(json_file) as f:
            results.extend(json.load(f))

    if not results:
        print(f"No result JSON files found in {eval_dir}/")
        return

    print(f"Loaded {len(results)} results from {eval_dir}/")
    run_views(results, evaluation_config, eval_dir=eval_dir)
    print("Done.")


if __name__ == "__main__":
    main()
'''


def render_algorithm_stub(algo_name: str) -> str:
    return f'''"""
Algorithm: {algo_name}

Implement your MAPF algorithm here.
The class must implement `act(observations)` and `reset_states()`.
Optionally implement `act_batch(observations, dones)` for GPU-batched evaluation.
"""


class MyAlgorithm:
    def __init__(self, **kwargs):
        """
        Initialize your algorithm.

        kwargs may include fields from your algorithm config in eval_config.yaml
        (e.g. device, seed, etc.)
        """
        pass

    def act(self, observations) -> list:
        """
        Compute actions for all agents given their observations.

        Args:
            observations: List of observation dicts, one per agent.
                          Each contains keys like 'obstacles', 'agents', 'xy', 'target_xy'.

        Returns:
            List of actions (ints), one per agent.
            Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        """
        # Example: all agents stay in place
        return [0] * len(observations)

    def reset_states(self):
        """Reset any internal state between episodes."""
        pass

    # --- Optional: implement for GPU-batched evaluation ---
    # def act_batch(self, observations, dones):
    #     """
    #     Batched version of act() for multiple environments.
    #
    #     Args:
    #         observations: List of List of observation dicts.
    #                       Outer list = environments, inner = agents.
    #         dones: List of bools indicating which environments are done.
    #
    #     Returns:
    #         List of List of actions. Outer = environments, inner = agents.
    #     """
    #     return [[0] * len(obs) for obs in observations]
'''
