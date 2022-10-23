import os
import shutil
from copy import copy
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, Tuple, Union

import git
import optuna
from torch.distributed.run import run

import pytorch_tao as tao
from pytorch_tao import core


class DirtyRepoError(Exception):
    """Raised if action need clean repo but it is not"""


class Repo:
    """Code base folder of Tao.
    A Repo is actually a git Repo that has a `.tao` folder which
    includes configuration in it.

    Args:
        path: path of the repo.
    """
    def __init__(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.name = path.name
        self.tao_path = path / ".tao"
        self.git_path = path / ".git"
        self.cfg_path = self.tao_path / "cfg.py"
        if self.tao_path.exists():
            self._preset()

    def _preset(self):
        self._load_git()
        self._load_cfg()

    def _load_cfg(self):
        tao.load_cfg(self.cfg_path)

    def _load_git(self):
        self.git = git.Repo(self.path)

    def init(self):
        """Making a existing folder a tao repo"""
        self.tao_path.mkdir(exist_ok=False)
        config_content = core.render_tpl(
            "cfg.py",
            name=self.name,
            path=self.path.resolve().as_posix(),
            run_dir=(self.tao_path / "runs").resolve().as_posix(),
            log_dir=(self.path / "log").resolve().as_posix(),
        )
        self.cfg_path.write_text(config_content)
        gitignore_content = core.render_tpl("gitignore")
        (self.path / ".gitignore").write_text(gitignore_content)

        self.git = git.Repo.init(self.path)
        self.git.git.add(all=True)
        self.git.index.commit("initial commit")
        self._load_cfg()

    def commit_all(self, message: str):
        """Commit all the dirty changes to git
        It is equal to :code:`git add -A; git commit -m xxx`

        Args:
            message: the message of `git commit`
        """
        self.git.git.add(all=True)
        self.git.index.commit(message)

    def exists(self) -> bool:
        """Is this tao repo exists and valid
        Returns:
            bool True exists False not
        """
        return self.path.exists() and self.tao_path.exists() and self.git_path.exists()

    def _torchrun(self, wd: Path, env: Dict[str, str], exclude_args: Tuple[str]):
        env = {k: v for k, v in env.items() if v is not None}
        try:
            prev_cwd = os.getcwd()
            os.chdir(wd)

            main_process_env = dict(os.environ)
            os.environ.update(env)
            args = copy(tao.args)
            for arg in exclude_args:
                delattr(args, arg)
            run(args)
        finally:
            os.chdir(prev_cwd)
            os.environ.clear()
            os.environ.update(main_process_env)

    def _prepare_wd(self):
        hexsha = self.git.head.ref.commit.hexsha[:8]
        wd = tao.cfg.run_dir / hexsha
        if not wd.exists():
            git.Repo.clone_from(self.path.as_posix(), wd.as_posix())
        return wd

    @core.ensure_config("run_dir", "study_storage")
    def tune(self):
        """Start hyperparameter tunning process.

        The `tao tune` interally call this function to start the tunning process.

        To call this function successfully, the config(:code:`tao.cfg`) should have
        "run_dir" and "study_storage" being set.

        raises:
            :class:`DirtyRepoError`: if repo is dirty
        """
        if self.git.is_dirty():
            raise DirtyRepoError("`tao tune` requires the repo to be clean")
        if tao.cfg.study_storage is None:
            raise ValueError("In memory study is not supported in tao")
        tao_name = f"tune_{self._get_tao_name()}"
        tao.study = optuna.create_study(
            study_name=tao_name,
            storage=tao.cfg.study_storage,
            load_if_exists=tao.args.tao_tune_duplicated,
            direction=tao.cfg.tune_direction,
        )
        wd = self._prepare_wd()
        for _ in range(tao.args.tao_tune_max_trials or int(1e9)):
            env = {
                "TAO_COMMIT": self.git.head.ref.commit.hexsha,
                "TAO_NAME": tao_name,
                "TAO_TUNE": "1",
                "TAO_RUN_AT": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "TAO_REPO": wd.as_posix(),
                "PYTHONPATH": f'{wd.as_posix()}:{os.getenv("PYTHONPATH", "")}',
            }
            self._torchrun(wd, env, ("tao_name", "tao_tune_duplicated"))

    def _get_tao_name(self):
        if tao.args.tao_name:
            return tao.args.tao_name
        name = datetime.now().strftime("%m-%d_%H:%M")
        if not self.is_dirty():
            name = f"{name}_{self.head_hexsha(short=True)}"
        return name

    def is_dirty(self) -> bool:
        return self.git.is_dirty()

    def head_hexsha(self, short=False) -> str:
        hexsha = self.git.head.ref.commit.hexsha
        if short:
            return hexsha[:8]
        return hexsha

    @core.ensure_config("run_dir")
    def run(self):
        """Start a training process.

        Run will call :code:`torch.distributed.run` so this func will rely on the
        command line arguments. Call this method with right command line options.

        raises:
            :class:`DirtyRepoError`: if --dirty is not passed and the repo is dirty
        """
        if tao.args.tao_commit:
            self.git.git.add(all=True)
            self.git.index.commit(tao.args.tao_commit)
        if not tao.args.tao_dirty and self.git.is_dirty(untracked_files=True):
            raise DirtyRepoError(
                "`tao run` requires the repo to be clean, "
                "or use `tao run --dirty` to run in dirty mode"
            )
        wd = self.path if tao.args.tao_dirty else self._prepare_wd()
        env = {
            "TAO_NAME": self._get_tao_name(),
            "TAO_COMMIT": self.git.head.ref.commit.hexsha,
            "TAO_RUN_AT": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "TAO_REPO": wd.as_posix(),
            "PYTHONPATH": f'{wd.as_posix()}:{os.getenv("PYTHONPATH", "")}',
            "TAO_DIRTY": "1" if self.is_dirty() else None,
        }
        self._torchrun(wd, env, ("tao_dirty", "tao_cmd", "tao_commit"))

    @classmethod
    def create(cls, path: Union[Path, str]) -> "Repo":
        """Create a tao project from scratch.
        :code:`tao new` internally call this method, it is equal to
        :code:`mkdir path; tao init path;`

        Args:
            path: an not exist path to create the `tao.repo`

        Returns:
            :class:`Repo`: a repo object represents the created one
        """
        path = Path(path)
        path.mkdir(exist_ok=False)
        repo = cls(path)
        repo.init()
        return repo

    @classmethod
    def find_by_file(cls, path: Union[Path, str]) -> "Repo":
        """Find the nearest tao repo of any file.
        When calling `tao run some_script.py`, tao will try to find the
        repo of some_script.py, the file can be nested in sub folders of
        a tao repo.

        Args:
            path: path of the file.

        Returns:
            :class:`Repo`: a repo that the file belongs to.

        Raises:
            FileNotFoundError: if no repos are found.
        """
        path = Path(path).resolve()
        while True:
            if path.as_posix() == "/":
                raise FileNotFoundError()
            if (path / ".tao").exists():
                break
            path = path.parent
        return cls(path)

    @core.ensure_config("kaggle_username", "kaggle_key", "kaggle_dataset_id")
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
