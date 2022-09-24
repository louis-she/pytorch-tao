import click

import pytorch_tao as tao


@click.group()
def main():
    pass


@main.command()
@click.argument("path")
def new(path):
    """Create a new project"""
    repo = tao.Repo(path)
    repo.create()


@main.command()
@click.option("tao_repo", default=".")
def sync_kaggle_dataset(tao_repo):
    repo = tao.Repo(tao_repo, exists=True)
    repo.create_sync_kaggle_dataset_action()


@main.command()
def run():
    print("call run")
