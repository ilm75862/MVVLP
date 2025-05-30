# Following PEP 517/518, this file should not not needed and replaced instead by the setup.cfg file and pyproject.toml.
# Unfortunately it is still required py the pip editable mode `pip install -e`
# See https://stackoverflow.com/a/60885212

import pathlib
from setuptools import setup, find_packages

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the gymnasium version."""
    path = CWD / "avp_env" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(name="avp-env",
      version=get_version(),
      packages=find_packages(include=["avp_env", "avp_env.*"]), )
