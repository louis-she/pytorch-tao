import os
import shutil
from argparse import ArgumentError
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, Tuple, Union

import git
import jinja2
import optuna
from filelock import FileLock
from torch.distributed.run import run

import pytorch_tao as tao
from pytorch_tao import core, exceptions


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
        self.lock = FileLock(self.tao_path / ".repo.lock")

    def _load_cfg(self):
        tao.load_cfg(self.cfg_path)

    def _load_git(self):
        self.git = git.Repo(self.path)

    def init(self, template: str):
        """Making a existing folder a tao repo"""
        try:
            self._create_proj_from_template(template)
        except jinja2.exceptions.TemplateNotFound:
            valid_templates = os.listdir(
                os.path.join(os.path.dirname(__file__), "templates", "projects")
            )
            valid_templates.remove("default")
            raise exceptions.TemplateNotFound(
                f"Template {template} not found, valid templates are {', '.join(valid_templates)}"
            )
        self.tao_path.mkdir(exist_ok=True)
        self.git = git.Repo.init(self.path)
        self.commit_all("initial commit")
        self._load_cfg()
        self.lock = FileLock(self.tao_path / ".repo.lock")

    def _create_proj_from_template(self, proj_name="mini"):
        self._create_proj_file(proj_name, ".gitignore")
        self._create_proj_file(
            proj_name,
            ".tao/cfg.py",
            name=self.name,
            path=self.path.resolve().as_posix(),
            run_dir=(self.tao_path / "runs").resolve().as_posix(),
            log_dir=(self.path / "log").resolve().as_posix(),
        )
        self._create_proj_file(proj_name, "main.py")

    def _create_proj_file(self, proj_name: str, file_name: str, **context_vars):
        file_path = self.path / file_name
        if (self.path / file_name).exists():
            return
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = core.render_tpl(
                f"projects/{proj_name}/{file_name}", **context_vars
            )
        except jinja2.exceptions.TemplateNotFound:
            content = core.render_tpl(f"projects/default/{file_name}", **context_vars)
        file_path.write_text(content)

    def commit_all(self, message: str) -> git.Commit:
        """Commit all the dirty changes to git
        It is equal to :code:`git add -A; git commit -m xxx`

        Args:
            message: the message of `git commit`
        """
        self.git.git.add(all=True)
        return self.git.index.commit(message)

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
            args = core.parse_tao_args()
            for arg in exclude_args:
                delattr(args, arg)
            run(args)
        finally:
            os.chdir(prev_cwd)
            os.environ.clear()
            os.environ.update(main_process_env)

    def _prepare_wd(self, checkout: str = None) -> Path:
        if checkout:
            try:
                current_pos = self.git.active_branch.name
            except TypeError:
                current_pos = self.git.rev_parse("HEAD")
            self.git.git.checkout(checkout)
            hexsha = self.git.rev_parse("HEAD").hexsha[:8]
            self.git.git.checkout(current_pos)
        else:
            hexsha = self.git.head.ref.commit.hexsha[:8]
        wd = tao.cfg.run_dir / hexsha
        if not wd.exists():
            git.Repo.clone_from(self.path.as_posix(), wd.as_posix())
            git.Repo(wd).git.checkout(hexsha)
        return wd

    @core.ensure_config("run_dir", "study_storage")
    def tune(self, name: str, max_trials: int, duplicated: bool):
        """Start hyperparameter tunning process.

        The `tao tune` interally call this function to start the tunning process.

        To call this function successfully, the config(:code:`tao.cfg`) should have
        "run_dir" and "study_storage" being set.

        raises:
            :class:`DirtyRepoError`: if repo is dirty
        """
        if self.git.is_dirty():
            raise exceptions.DirtyRepoError("`tao tune` requires the repo to be clean")
        if tao.cfg.study_storage is None:
            raise ValueError("In memory study is not supported in tao")
        if not name:
            name = self._gen_name()
        with self.lock:
            tao.study = optuna.create_study(
                study_name=name,
                storage=tao.cfg.study_storage,
                load_if_exists=duplicated,
                direction=tao.cfg.tune_direction,
            )
            wd = self._prepare_wd()
        for _ in range(max_trials or int(1e9)):
            env = {
                "TAO_COMMIT": self.git.head.ref.commit.hexsha,
                "TAO_NAME": name,
                "TAO_TUNE": "1",
                "TAO_RUN_AT": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "TAO_REPO": wd.as_posix(),
                "PYTHONPATH": f'{wd.as_posix()}:{os.getenv("PYTHONPATH", "")}',
            }
            self._torchrun(wd, env, ("tao_tune_name", "tao_tune_duplicated"))

    def _gen_name(self) -> str:
        if self.is_dirty():
            return "dirty"
        return "unnamed"

    def is_dirty(self) -> bool:
        return self.git.is_dirty()

    def head_hexsha(self, short=False) -> str:
        try:
            hexsha = self.git.head.ref.commit.hexsha
        except TypeError:
            hexsha = self.git.rev_parse("HEAD").hexsha
        if short:
            return hexsha[:8]
        return hexsha

    @core.ensure_config("run_dir")
    def run(self, name: str, dirty: bool, checkout: "str"):
        """Start a training process.

        Run will call :code:`torch.distributed.run` so this func will rely on the
        command line arguments. Call this method with right command line options.

        raises:
            :class:`.DirtyRepoError`: if --dirty is not passed and the repo is dirty
        """
        if checkout and dirty:
            raise ArgumentError("checkout and dirty can not be used at the same time")

        if not checkout and not dirty and self.git.is_dirty(untracked_files=True):
            raise exceptions.DirtyRepoError(
                "`tao run` requires the repo to be clean, "
                "or use `tao run --dirty` to run in dirty mode"
            )

        with self.lock:
            wd = self.path if dirty else self._prepare_wd(checkout)
            env = {
                "TAO_NAME": name if name else self._gen_name(),
                "TAO_COMMIT": self.git.head.ref.commit.hexsha,
                "TAO_RUN_AT": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "TAO_REPO": wd.as_posix(),
                "PYTHONPATH": f'{wd.as_posix()}:{os.getenv("PYTHONPATH", "")}',
                "TAO_DIRTY": "1" if self.is_dirty() else None,
            }
        self._torchrun(wd, env, ("tao_dirty", "tao_cmd", "tao_checkout"))

    @classmethod
    def create(cls, path: Union[Path, str], template: str = "mini") -> "Repo":
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
        repo.init(template)
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
