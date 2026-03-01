import typer
from rich.console import Console
from rich.theme import Theme

# POGEMA color palette (from SvgSettings)
POGEMA_THEME = Theme({
    "pogema.red": "#c1433c",       # ego_color — primary accent, errors
    "pogema.blue": "#2e6f9e",      # — commands, paths
    "pogema.muted": "#6e81af",     # ego_other_color — info text
    "pogema.obstacle": "#84A1AE",  # obstacle_color — borders, panels
    "pogema.teal": "#00b9c8",      # — highlights
    "pogema.light": "#72D5C8",     # — secondary highlights
    "pogema.green": "#0ea08c",     # — success/done
    "pogema.warm": "#8F7B66",      # — warnings
})

app = typer.Typer(
    name="ptb",
    help="POGEMA Toolbox CLI — scaffold, run, and visualize MAPF algorithm evaluations.",
    no_args_is_help=True,
)
console = Console(theme=POGEMA_THEME)


@app.callback()
def main():
    """POGEMA Toolbox — evaluation framework for Multi-Agent Path Finding algorithms."""


from pogema_toolbox.cli.cmd_init import init  # noqa: E402
from pogema_toolbox.cli.cmd_run import run  # noqa: E402
from pogema_toolbox.cli.cmd_render import render  # noqa: E402
from pogema_toolbox.cli.cmd_check import check  # noqa: E402

app.command("init")(init)
app.command("run")(run)
app.command("render")(render)
app.command("check")(check)


def cli_main():
    app()
