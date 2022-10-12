import os

import click


@click.group()
def cli():
    pass


@cli.command()
def format():
    os.system("black .")
    os.system("usort format .")
    os.system(
        "flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics"
    )


def main():
    cli()
