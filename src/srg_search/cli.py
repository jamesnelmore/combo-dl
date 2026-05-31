import typer

from srg_search.cayley import cli as cayley_cli

app = typer.Typer()

app.add_typer(cayley_cli.app, name="cayley")


@app.command()
def hello(name: str):
    print(f"Hello {name}")


if __name__ == "__main__":
    app()
