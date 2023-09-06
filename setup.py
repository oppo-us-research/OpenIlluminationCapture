from setuptools import find_packages
from setuptools import setup

setup(
    name="ltsg",
    version="0.1",
    author="Isabella Liu, Linghao Chen",
    packages=find_packages(exclude=("configs", "tests", "models", "data", "dbg")),
)
