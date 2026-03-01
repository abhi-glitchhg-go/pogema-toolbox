from pathlib import Path

import questionary
import typer
import yaml
from rich.panel import Panel
from rich.syntax import Syntax

from pogema_toolbox.cli.app import console
from pogema_toolbox.cli.templates import (
    render_eval_config,
    render_run_script,
    render_render_script,
    render_algorithm_stub,
)

PARALLEL_BACKENDS = [
    "sequential",
    "balanced_multiprocessing",
    "balanced_dask",
    "multiprocessing",
    "dask",
    "batched",
]


def _ask_or_abort(result):
    if result is None:
        raise typer.Abort()
    return result


def init(
    output_dir: str = typer.Argument(
        None,
        help="Directory to scaffold the evaluation project in. Prompted if not given.",
    ),
):
    """Interactively scaffold a new POGEMA evaluation project."""

    console.print(Panel(
        "[bold]POGEMA Toolbox — Project Scaffolding Wizard[/bold]",
        border_style="pogema.obstacle",
        expand=False,
    ))

    # --- Project directory ---
    if output_dir is None:
        output_dir = _ask_or_abort(
            questionary.text(
                "Project directory name:",
                default="my_eval",
            ).ask()
        )
    project_dir = Path(output_dir)

    if project_dir.exists() and any(project_dir.iterdir()):
        overwrite = questionary.confirm(
            f"Directory '{project_dir}' already exists and is not empty. Continue anyway?",
            default=False,
        ).ask()
        if not overwrite:
            raise typer.Abort()

    # --- Algorithm setup ---
    BUILTIN_ALGORITHMS = {
        "A* (BatchAStarAgent from pogema)": {
            "import": "pogema:BatchAStarAgent",
            "name": "A*",
        },
        "A* (AStarAgent from pogema)": {
            "import": "pogema:AStarAgent",
            "name": "A*",
        },
        "MAPF-GPT (requires mapf-gpt package)": {
            "import": "mapf_gpt.inference:MAPFGPTInference",
            "config_import": "mapf_gpt.inference:MAPFGPTInferenceConfig",
            "name": "MAPF-GPT",
        },
    }

    algo_mode = _ask_or_abort(
        questionary.select(
            "How do you want to include your algorithm?",
            choices=[
                questionary.Choice("Use a built-in algorithm (recommended)", value="builtin"),
                questionary.Choice("Generate a template algorithm file", value="template"),
                questionary.Choice("Use an existing class (import path)", value="import"),
            ],
        ).ask()
    )

    algo_import_path = None
    algo_config_import_path = None

    if algo_mode == "builtin":
        algo_builtin_key = _ask_or_abort(
            questionary.select(
                "Which built-in algorithm?",
                choices=list(BUILTIN_ALGORITHMS.keys()),
            ).ask()
        )
        builtin = BUILTIN_ALGORITHMS[algo_builtin_key]
        algo_import_path = builtin["import"]
        algo_config_import_path = builtin.get("config_import")
        algo_name = builtin["name"]
    else:
        algo_name = _ask_or_abort(
            questionary.text("Algorithm display name:", default="MyAlgorithm").ask()
        )

        if algo_mode == "import":
            algo_import_path = _ask_or_abort(
                questionary.text(
                    "Import path (module:ClassName):",
                    default="my_module:MyAgent",
                    instruction="e.g. my_agents.astar:AStarAgent",
                ).ask()
            )

    # --- Environment settings ---
    console.print("\n[bold pogema.teal]Environment Configuration[/bold pogema.teal]")

    agents_input = _ask_or_abort(
        questionary.text(
            "Number of agents (comma-separated for grid search):",
            default="8, 16, 32",
            instruction="e.g. 8, 16, 32",
        ).ask()
    )
    num_agents = [int(x.strip()) for x in agents_input.split(",")]

    num_seeds = _ask_or_abort(
        questionary.text(
            "Number of random seeds:",
            default="3",
            validate=lambda v: v.isdigit() or "Must be a number",
        ).ask()
    )
    num_seeds = int(num_seeds)
    seeds = list(range(num_seeds))

    # --- Parallel backend ---
    parallel_backend = _ask_or_abort(
        questionary.select(
            "Parallel backend:",
            choices=PARALLEL_BACKENDS,
            default="sequential",
        ).ask()
    )

    num_process = 1
    if parallel_backend not in ("sequential", "batched"):
        num_process_input = _ask_or_abort(
            questionary.text(
                "Number of parallel processes:",
                default="4",
                validate=lambda v: v.isdigit() or "Must be a number",
            ).ask()
        )
        num_process = int(num_process_input)

    # --- Visualization ---
    console.print("\n[bold pogema.teal]Result Views[/bold pogema.teal]")
    views = _ask_or_abort(
        questionary.checkbox(
            "Which result views to include?",
            choices=[
                questionary.Choice("Tabular summary", value="tabular", checked=True),
                questionary.Choice("Line plot", value="plot"),
                questionary.Choice("Multi-panel plot", value="multi-plot"),
            ],
        ).ask()
    )

    # --- Build config ---
    env_config = {
        "name": "Pogema-v0",
        "observation_type": "MAPF",
        "on_target": "restart",
        "max_episode_steps": 32,
        "density": 0.3,
        "size": 32,
        "num_agents": {"grid_search": num_agents} if len(num_agents) > 1 else num_agents[0],
        "seed": {"grid_search": seeds} if len(seeds) > 1 else seeds[0],
        "use_maps": False,
    }

    is_mapf_gpt = algo_mode == "builtin" and "MAPF-GPT" in algo_name

    if is_mapf_gpt:
        algo_base = {
            "name": algo_name,
            "parallel_backend": parallel_backend,
        }
        if num_process > 1:
            algo_base["num_process"] = num_process
        algorithms_config = {
            "MAPF-GPT-2M": {**algo_base, "path_to_weights": "weights/model-2M.pt"},
        }
        algo_key = "MAPF-GPT-2M"
    else:
        algo_config = {
            "name": algo_name,
            "parallel_backend": parallel_backend,
        }
        if num_process > 1:
            algo_config["num_process"] = num_process
        algo_key = algo_name.replace(" ", "-")
        algorithms_config = {algo_key: algo_config}

    views_config = _build_views_config(views, num_agents, seeds)

    full_config = {
        "environment": env_config,
        "algorithms": algorithms_config,
    }
    if views_config:
        full_config["results_views"] = views_config

    # --- Write files ---
    project_dir.mkdir(parents=True, exist_ok=True)

    config_yaml = render_eval_config(full_config)
    if is_mapf_gpt:
        config_yaml = _append_mapf_gpt_comments(config_yaml)
    config_path = project_dir / "eval_config.yaml"
    config_path.write_text(config_yaml)

    effective_mode = "import" if algo_mode == "builtin" else algo_mode
    run_script = render_run_script(
        algo_name=algo_name,
        algo_key=algo_key,
        algo_mode=effective_mode,
        algo_import_path=algo_import_path,
        algo_config_import_path=algo_config_import_path,
    )
    (project_dir / "run_eval.py").write_text(run_script)

    render_script = render_render_script()
    (project_dir / "render_results.py").write_text(render_script)

    if effective_mode == "template":
        algo_stub = render_algorithm_stub(algo_name)
        (project_dir / "algorithm.py").write_text(algo_stub)

    # --- Summary ---
    console.print()
    console.print(Panel(
        f"[bold pogema.green]Project scaffolded in [white]{project_dir}/[/white][/bold pogema.green]",
        border_style="pogema.green",
        expand=False,
    ))

    files_created = [
        f"  {config_path.name}        — evaluation configuration",
        f"  run_eval.py        — run the evaluation",
        f"  render_results.py  — re-render views from saved JSON results",
    ]
    if effective_mode == "template":
        files_created.append(f"  algorithm.py       — your algorithm implementation (fill in!)")

    console.print("[bold pogema.teal]Files created:[/bold pogema.teal]")
    for f in files_created:
        console.print(f"  {f}")

    console.print()
    console.print("[bold pogema.teal]Next steps:[/bold pogema.teal]")
    if effective_mode == "template":
        console.print(f"  1. Implement your algorithm in [pogema.blue]{project_dir}/algorithm.py[/pogema.blue]")
        console.print(f"  2. Run: [pogema.blue]cd {project_dir} && python run_eval.py[/pogema.blue]")
    else:
        console.print(f"  1. Run: [pogema.blue]cd {project_dir} && python run_eval.py[/pogema.blue]")
    console.print(f"  Re-render results: [pogema.blue]python render_results.py[/pogema.blue]")

    console.print()
    console.print(Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False))


def _build_views_config(views, num_agents, seeds):
    config = {}

    if "tabular" in views:
        config["TabularView"] = {
            "type": "tabular",
            "drop_keys": ["seed"],
            "print_results": True,
        }

    if "plot" in views:
        plot_cfg = {
            "type": "plot",
            "x": "num_agents",
            "y": "ISR",
            "by": "algorithm",
        }
        if len(seeds) > 1:
            plot_cfg["error_bar"] = "ci"
        config["PlotView"] = plot_cfg

    if "multi-plot" in views:
        config["MultiPlotView"] = {
            "type": "multi-plot",
            "x": "num_agents",
            "y": "ISR",
            "by": "algorithm",
            "over": "map_name",
            "num_cols": 2,
        }

    return config


def _append_mapf_gpt_comments(config_yaml: str) -> str:
    """Append commented-out MAPF-GPT model variants after the active algorithm entry."""
    # Find the MAPF-GPT-2M block and capture its indentation + fields to replicate
    lines = config_yaml.splitlines(keepends=True)
    result = []

    # Find the line with path_to_weights to know when the active entry ends
    i = 0
    insert_idx = None
    indent = "    "
    while i < len(lines):
        result.append(lines[i])
        if "path_to_weights:" in lines[i] and insert_idx is None:
            insert_idx = len(result)
            # Detect indentation of the algo key (2 levels up)
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            # algo key indent is one level less
            key_indent = indent.rstrip()
            if len(indent) >= 2:
                key_indent = indent[:len(indent)//2] if indent[0] == ' ' else ''
        i += 1

    if insert_idx is None:
        return config_yaml

    # Build commented-out entries
    commented = _mapf_gpt_commented_entries(indent)
    result.insert(insert_idx, commented)

    return "".join(result)


def _mapf_gpt_commented_entries(field_indent: str) -> str:
    """Generate commented-out MAPF-GPT algorithm entries."""
    # key_indent is one level above field_indent
    if len(field_indent) >= 2:
        key_indent = field_indent[: len(field_indent) // 2]
    else:
        key_indent = ""

    models = [
        ("MAPF-GPT-2M-DDG", "weights/model-2M-DDG.pt", "2M params, DDG variant"),
        ("MAPF-GPT-6M", "weights/model-6M.pt", "6M params"),
        ("MAPF-GPT-85M", "weights/model-85M.pt", "85M params, largest"),
    ]

    lines = [f"{key_indent}# Uncomment to add more MAPF-GPT model variants:\n"]
    for name, weights, desc in models:
        lines.append(f"{key_indent}# {name}:  # {desc}\n")
        lines.append(f"{key_indent}#   name: MAPF-GPT\n")
        lines.append(f"{key_indent}#   parallel_backend: sequential\n")
        lines.append(f"{key_indent}#   path_to_weights: {weights}\n")

    return "".join(lines)
