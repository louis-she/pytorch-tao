import logging
import os
import shutil
from copy import copy
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Union

import git
import optuna
from torch.distributed.run import run

import pytorch_tao as tao
from pytorch_tao import core


class DirtyRepoError(Exception):
    """Raised if action need clean repo but it is not"""


class Repo:
    def __init__(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.name = path.name
        self.tao_path = path / ".tao"
        self.git_path = path / ".git"
        self.cfg_path = self.tao_path / "cfg.py"
        if self.tao_path.exists():
            try:
                self.git = git.Repo(self.path)
            except git.InvalidGitRepositoryError:
                logging.warning("failed to initialze git, is .git dir in tao repo?")
            try:
                self.load_cfg()
            except FileNotFoundError:
                logging.warning("failed to load config in tao repo")

    def init(self):
        self.tao_path.mkdir(exist_ok=False)
        config_content = core.render_tpl(
            "cfg.py",
            name=self.name,
            run_dir=(self.tao_path / "runs").resolve().as_posix(),
        )
        self.cfg_path.write_text(config_content)
        gitignore_content = core.render_tpl(".gitignore")
        (self.path / ".gitignore").write_text(gitignore_content)

        self.git = git.Repo.init(self.path)
        self.git.git.add(all=True)
        self.git.index.commit("initial commit")
        self.load_cfg()

    def load_cfg(self):
        tao.load_cfg(self.cfg_path)

    def commit_all(self, message: str):
        self.git.git.add(all=True)
        self.git.index.commit(message)

    def exists(self):
        """Is this tao repo exists and valid"""
        return self.path.exists() and self.tao_path.exists() and self.git_path.exists()

    @tao.ensure_config("run_dir", "study_storage")
    def tune(self):
        """Start hyperparameter tunning process"""
        if self.git.is_dirty():
            raise DirtyRepoError()
        if tao.cfg.study_storage is None:
            raise ValueError("In memory study is not supported in tao")
        tao.args.tao_commit = False
        tao.args.tao_dirty = False
        tao.study = optuna.create_study(
            study_name=tao.args.tao_tune_name,
            storage=tao.cfg.study_storage,
            load_if_exists=tao.args.tao_tune_duplicated,
            direction=tao.cfg.tune_direction,
        )
        os.environ["TAO_TUNE"] = tao.args.tao_tune_name
        for i in range(tao.args.tao_tune_max_trials):
            logging.info(f"tune {tao.args.tao_tune_name} the {i+1} time")
            self.run()
            logging.info(f"tune {tao.args.tao_tune_name} {i+1} has finished")

    @tao.ensure_config("run_dir")
    def run(self):
        """Start a training process.

        Run will call torch.distributed.run.run so this func will rely on the
        command line arguments. Call this function with right command line options.
        """
        if tao.args.tao_commit:
            self.git.git.add(all=True)
            self.git.index.commit(tao.args.tao_commit)
        if not tao.args.tao_dirty and self.git.is_dirty(untracked_files=True):
            raise DirtyRepoError()
        run_dir = Path(tao.cfg.run_dir)
        run_dir = run_dir if run_dir.is_absolute() else self.path / run_dir
        metadata = {
            "dirty": self.git.is_dirty(untracked_files=True),
            "commit": self.git.head.ref.commit.hexsha,
            "run_at": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        }
        if tao.args.tao_dirty:
            # run process in current pwd
            run_fold = self.path
        else:
            # run process in run dir
            hexsha = self.git.head.ref.commit.hexsha[:8]
            run_fold = run_dir / hexsha
            if not run_fold.exists():
                git.Repo.clone_from(self.path.as_posix(), run_fold.as_posix())
        args = copy(tao.args)
        del args.tao_dirty
        del args.tao_cmd
        del args.tao_commit
        for key, val in metadata.items():
            if isinstance(val, bool):
                args.training_script_args += [f"--{key}"] if val else []
                continue
            args.training_script_args += [f"--{key}", val]
        prev_cwd = os.getcwd()
        os.chdir(run_fold)
        os.environ[
            "PYTHONPATH"
        ] = f'{run_fold.as_posix()}:{os.getenv("PYTHONPATH", "")}'
        os.environ["TAO_REPO"] = run_fold.as_posix()
        run(args)
        os.chdir(prev_cwd)

    @classmethod
    def create(cls, path: Union[Path, str]):
        """Create a tao project from scratch"""
        path = Path(path)
        path.mkdir(exist_ok=False)
        repo = cls(path)
        repo.init()
        return repo

    @classmethod
    def find_by_file(cls, path: Union[Path, str]) -> "Repo":
        """Find the nearest tao repo of any file"""
        path = Path(path).resolve()
        while True:
            if path.as_posix() == "/":
                raise FileNotFoundError()
            if (path / ".tao").exists():
                break
            path = path.parent
        return cls(path)

    @tao.ensure_config("kaggle_username", "kaggle_key", "kaggle_dataset_id")
    def sync_code_to_kaggle(self):
        """Create a GitHub workflow that sync the source code to Kaggle dataset.

        There are presteps before generating this action:
        1. Create a dataset with any file(which will give the `dataset_slug` parameter)
        2. In the GitHub repo settings, add two GitHub action secrets named KAGGLE_USERNAME and KAGGLE_KEY
        """

        # kaggle do authentication when importing the package
        # so the import goes here
        os.environ["KAGGLE_USERNAME"] = tao.cfg.kaggle_username
        os.environ["KAGGLE_KEY"] = tao.cfg.kaggle_key
        import kaggle

        tempdir = Path(mkdtemp())
        output_file = tempdir / "output.zip"
        with output_file.open("wb") as f:
            self.git.archive(f, format="zip")
        metadata_file = tempdir / "dataset-metadata.json"
        metadata = """{
  "id": "%s",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
""" % (
            tao.cfg.kaggle_dataset_id,
        )
        metadata_file.write_text(metadata)
        kaggle.api.dataset_create_version(
            folder=tempdir, version_notes=self.git.head.ref.commit.message
        )
        shutil.rmtree(tempdir)
