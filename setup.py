from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="tetris_ai",
    version="0.0.1",
    install_requires=required,
)
