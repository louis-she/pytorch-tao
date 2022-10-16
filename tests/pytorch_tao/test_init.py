import pytorch_tao as tao


def test_version():
    assert tao.__version__ is not None


def test_global_attr_are_set():
    assert hasattr(tao, "args")
    assert hasattr(tao, "study")
    assert hasattr(tao, "trial")
    assert hasattr(tao, "cfg")
    assert hasattr(tao, "repo")
    assert hasattr(tao, "tracker")
    assert hasattr(tao, "log_dir")
    assert hasattr(tao, "name")
    assert hasattr(tao, "tune")
