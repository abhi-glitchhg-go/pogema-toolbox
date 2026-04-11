import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from pogema_toolbox.cli.app import console


def _format_set(values):
    """Format a set of values as a compact string."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return "-"
    if len(values) <= 3:
        return ", ".join(str(v) for v in values)
    return f"{values[0]}, ..., {values[-1]} ({len(values)})"


def _render_summary_table(check_result):
    """Compact per-algorithm summary table."""
    seed_data = check_result['seed_data']
    algorithms = check_result['algorithms']

    # Gather per-algorithm stats
    algo_stats = defaultdict(lambda: {'runs': 0, 'seeds': set(), 'maps': set(), 'agents': set(), 'duplicates': 0})
    for (map_name, num_agents, seed), algos in seed_data.items():
        for algo in algorithms:
            stats = algo_stats[algo]
            if algo in algos:
                count = len(algos[algo])
                stats['runs'] += count
                stats['seeds'].add(seed)
                stats['maps'].add(map_name)
                stats['agents'].add(num_agents)
                if count > 1:
                    stats['duplicates'] += count - 1

    # Count missing per algorithm
    for algo in algorithms:
        missing = 0
        for (_map_name, _num_agents, _seed), algos in seed_data.items():
            if algo not in algos:
                missing += 1
        algo_stats[algo]['missing'] = missing

    table = Table(border_style="pogema.obstacle", show_lines=True)
    table.add_column("Algorithm", style="bold")
    table.add_column("Runs", justify="right")
    table.add_column("Seeds", justify="right")
    table.add_column("Maps")
    table.add_column("Agents")
    table.add_column("Status")

    for algo in algorithms:
        stats = algo_stats[algo]
        issues = []
        if stats['missing'] > 0:
            issues.append(f"{stats['missing']} missing")
        if stats['duplicates'] > 0:
            issues.append(f"{stats['duplicates']} duplicates")

        if issues:
            status = f"[pogema.red]{', '.join(issues)}[/pogema.red]"
        else:
            status = "[pogema.green]ok[/pogema.green]"

        table.add_row(
            algo,
            str(stats['runs']),
            _format_set(stats['seeds']),
            _format_set(stats['maps']),
            _format_set(stats['agents']),
            status,
        )

    console.print(table)


def _render_verbose_table(check_result):
    """Full grid table with per-cell ok/missing/xN status."""
    seed_data = check_result['seed_data']
    algorithms = check_result['algorithms']

    table = Table(border_style="pogema.obstacle", show_lines=True)
    table.add_column("Map Name")
    table.add_column("Agents", justify="right")
    table.add_column("Seed", justify="right")
    for algo in algorithms:
        table.add_column(algo, justify="center")

    for (map_name, num_agents, seed) in sorted(seed_data.keys()):
        algos = seed_data[(map_name, num_agents, seed)]
        cells = []
        for algo in algorithms:
            if algo not in algos:
                cells.append("[pogema.red]missing[/pogema.red]")
            elif len(algos[algo]) > 1:
                cells.append(f"[pogema.warm]x{len(algos[algo])}[/pogema.warm]")
            else:
                cells.append("[pogema.green]ok[/pogema.green]")

        table.add_row(str(map_name), str(num_agents), str(seed), *cells)

    console.print(table)


def check(
    project_dir: str = typer.Argument(
        ".",
        help="Path to the evaluation project directory.",
    ),
    results_dir: str = typer.Option(
        "results",
        "--results-dir", "-r",
        help="Results subdirectory name.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show full grid table with per-seed status.",
    ),
):
    """Check seed consistency across evaluation results."""
    project_path = Path(project_dir)
    eval_dir = project_path / results_dir

    if not eval_dir.exists():
        console.print(f"[pogema.red]Results directory not found:[/pogema.red] {eval_dir}")
        raise typer.Exit(1)

    results = []
    for json_file in sorted(eval_dir.glob("*.json")):
        with open(json_file) as f:
            results.extend(json.load(f))

    if not results:
        console.print(f"[pogema.warm]No result JSON files found in {eval_dir}/[/pogema.warm]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Checking seed consistency in [pogema.blue]{eval_dir}[/pogema.blue] ({len(results)} results)[/bold]",
        border_style="pogema.obstacle",
        expand=False,
    ))

    from pogema_toolbox.views.view_utils import check_seeds
    check_result = check_seeds(results)

    for error in check_result['errors']:
        console.print(f"[pogema.red]Error:[/pogema.red] {error}")

    if not check_result['algorithms']:
        console.print("[pogema.warm]No valid results to check.[/pogema.warm]")
        raise typer.Exit(1)

    if verbose:
        _render_verbose_table(check_result)
    else:
        _render_summary_table(check_result)

    problem_count = check_result['problem_count']
    if problem_count > 0:
        console.print(f"\n[pogema.red]Detected {problem_count} problem(s).[/pogema.red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold pogema.green]Passed seed consistency check.[/bold pogema.green]")
