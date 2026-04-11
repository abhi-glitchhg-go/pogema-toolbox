import subprocess
import sys
from pathlib import Path

import typer
from rich.panel import Panel

from pogema_toolbox.cli.app import console


def run(
    project_dir: str = typer.Argument(
        ".",
        help="Path to the evaluation project directory (containing run_eval.py).",
    ),
    script: str = typer.Option(
        "run_eval.py",
        "--script", "-s",
        help="Evaluation script filename (relative to project dir).",
    ),
):
    """Run an evaluation by executing the project's run_eval.py script."""
    project_path = Path(project_dir)
    script_path = project_path / script

    if not script_path.exists():
        console.print(f"[pogema.red]Script not found:[/pogema.red] {script_path}")
        console.print("Did you run [pogema.blue]ptb init[/pogema.blue] first?")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Running [pogema.blue]{script_path}[/pogema.blue][/bold]",
        border_style="pogema.obstacle",
        expand=False,
    ))

    result = subprocess.run(
        [sys.executable, str(script_path.resolve())],
        cwd=str(project_path.resolve()),
    )
    raise typer.Exit(result.returncode)
