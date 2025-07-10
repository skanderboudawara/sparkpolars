"""Setup script for sparkpolars package."""

from pathlib import Path

from setuptools import setup

here = Path(__file__).resolve().parent

long_description = (here / "readme.md").read_text(encoding="utf-8")

setup(
    name="sparkpolars",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
