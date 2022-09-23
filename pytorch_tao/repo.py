from genericpath import isdir
from pathlib import Path
import re
import shutil
import kaggle
from tempfile import mkdtemp, tempdir
from typing import Union
import git


class Repo:
    def __init__(self, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.tao_path = path / ".tao"
        self.git_path = path / ".git"
        if not self.exists():
            raise FileNotFoundError()
        self.git = git.Repo(self.path)

    def exists(self):
        return self.path.exists() and self.tao_path.exists() and self.git_path.exists()

    @classmethod
    def create(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(exist_ok=False)
        git.Repo.init(path)
        (path / ".tao").mkdir(exist_ok=False)
        return cls(path)

    def sync_code_to_kaggle(self, dataset_id: str, title: str = None):
        """Create a GitHub workflow that sync the source code to Kaggle dataset.

        There are presteps before generating this action:
        1. Create a dataset with any file(which will give the `dataset_slug` parameter)
        2. In the GitHub repo settings, add two GitHub action secrets named KAGGLE_USERNAME and KAGGLE_KEY
        """
        if title is None:
            title = re.sub("/|-", "_", "dataset_id")
        tempdir = Path(mkdtemp())
        output_file = tempdir / "output.zip"
        with output_file.open("wb") as f:
            self.git.archive(f, format="zip")
        metadata_file = tempdir / "dataset-metadata.json"
        metadata = """{
  "title": "%s",
  "id": "%s",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
""" % (
            title,
            dataset_id,
        )
        metadata_file.write_text(metadata)
        kaggle.api.dataset_create_version(
            folder=tempdir, version_notes=self.git.head.ref.message
        )
        shutil.rmtree(tempdir)
