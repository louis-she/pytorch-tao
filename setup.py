import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read("readme.md").replace(
    'src="assets/',
    'src="https://raw.githubusercontent.com/louis-she/pytorch-tao/master/assets/',
)

VERSION = find_version("pytorch_tao", "__init__.py")

requirements = [
    "torch~=1.10",
    "GitPython~=3.1.27",
    "Jinja2~=3.1.2",
    "kaggle~=1.5.12",
    "optuna~=3.0.2",
    "pytorch-ignite~=0.4.10",
]

setup(
    name="pytorch-tao",
    version=VERSION,
    author="Chenglu She",
    python_requires=">3.8",
    author_email="chenglu.she@gmail.com",
    url="https://github.com/louis-she/pytorch-tao",
    description="A toolbox for a specific Machine Learning training project",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "tao = pytorch_tao.cli:main",
            "tao_devtool = pytorch_tao.devtool:main",
        ],
    },
)
