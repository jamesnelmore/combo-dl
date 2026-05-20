import typer

from .run_single import run_single

app = typer.Typer()

# TODO Various ways to call the CLI
# Can pass multiple param sets and specify the groups to check


@app.command()
def search(n: int, k: int, t: int, lambda_param: int, mu: int, group: int):
    pass  # TODO add support for list of param sets or list of groups. Group is optional and all of order n are searched if one is not provided


app.command(name="single")(run_single)


@app.command()
def cleanup():
    pass  # TODO string together analysis and deduplication scripts


if __name__ == "__main__":
    app()
