import json
from pathlib import Path

import typer
import yaml
from rich.panel import Panel

from pogema_toolbox.cli.app import console


def render(
    project_dir: str = typer.Argument(
        ".",
        help="Path to the evaluation project directory.",
    ),
    config: str = typer.Option(
        "eval_config.yaml",
        "--config", "-c",
        help="Config filename (relative to project dir).",
    ),
    results_dir: str = typer.Option(
        "results",
        "--results-dir", "-r",
        help="Results subdirectory name.",
    ),
):
    """Re-render result views from saved JSON files."""
    project_path = Path(project_dir)
    config_path = project_path / config
    eval_dir = project_path / results_dir

    if not config_path.exists():
        console.print(f"[pogema.red]Config not found:[/pogema.red] {config_path}")
        raise typer.Exit(1)

    if not eval_dir.exists():
        console.print(f"[pogema.red]Results directory not found:[/pogema.red] {eval_dir}")
        raise typer.Exit(1)

    with open(config_path) as f:
        evaluation_config = yaml.safe_load(f)

    results = []
    for json_file in sorted(eval_dir.glob("*.json")):
        with open(json_file) as f:
            results.extend(json.load(f))

    if not results:
        console.print(f"[pogema.warm]No result JSON files found in {eval_dir}/[/pogema.warm]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Re-rendering views from [pogema.blue]{eval_dir}[/pogema.blue] ({len(results)} results)[/bold]",
        border_style="pogema.obstacle",
        expand=False,
    ))

    from pogema_toolbox.evaluator import run_views
    run_views(results, evaluation_config, eval_dir=eval_dir)

    console.print("[bold pogema.green]Done.[/bold pogema.green]")
