import os

import pytest
import pytorch_tao as tao


def test_ensure_config(test_repo: tao.Repo):
    @tao.ensure_config("mount_drive", "dataset_dir")
    def read_colab_drive_file():
        return True

    with pytest.raises(tao.ConfigMissingError) as e:
        read_colab_drive_file()
        assert e.missing_keys == {"mount_drive"}
        assert e.func.__name__ == "read_colab_drive_file"

    os.environ["TAO_ENV"] = "colab"
    tao.load_cfg(test_repo.cfg_path)
    assert read_colab_drive_file()
