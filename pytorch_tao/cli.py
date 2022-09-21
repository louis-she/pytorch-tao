import click


@click.group()
def main():
    pass


@main.command()
def run():
    print("call run")
