import os
import re
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Union

import git

import pytorch_tao as tao


class Repo:
    def __init__(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.tao_path = path / ".tao"
        self.git_path = path / ".git"
        self.cfg_path = self.tao_path / "cfg.yml"
        if not self.exists():
            raise FileNotFoundError()
        self.git = git.Repo(self.path)
        tao.load_cfg(self.cfg_path)

    def exists(self):
        return self.path.exists() and self.tao_path.exists() and self.git_path.exists()

    @classmethod
    def create(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(exist_ok=False)
        git.Repo.init(path)
        (path / ".tao").mkdir(exist_ok=False)
        (path / ".tao" / "cfg.yml").write_text(
            """# config file of tao
        """
        )
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
        os.environ["KAGGLE_USERNAME"] = tao.cfg["kaggle_username"]
        os.environ["KAGGLE_KEY"] = tao.cfg["kaggle_key"]
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
            tao.cfg["kaggle_dataset_id"],
        )
        metadata_file.write_text(metadata)
        kaggle.api.dataset_create_version(
            folder=tempdir, version_notes=self.git.head.ref.message
        )
        shutil.rmtree(tempdir)
