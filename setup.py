"""Setups the project."""
import itertools
import re

from setuptools import find_packages, setup


print("Checking version...")
with open("mdgym/version.py") as file:
    full_version = file.read()
    matched_version = re.match(r'VERSION = "\d\.\d+\.\d+"', full_version).group()
    assert (matched_version == full_version), "Unexpected version: {}".format(full_version)
    VERSION = re.search(r"\d\.\d+\.\d+", full_version).group()

# Environment-specific dependencies.
extras = {
    "hawkins2DMap": [""],
}

# Testing dependency groups.
# None

# Uses the readme as the description on PyPI
with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

setup(
    author="Aditya Rauniyar",
    author_email="rauniyar@cmu.edu",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Gym environment for drone with camera coverage that supports multiple drones and actors on the "
                "environment.",
    extras_require=extras,
    install_requires=[
        "numpy >= 1.18.0",
        "open3d == 0.17.0",

    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="multi-drone-mdgym",
    packages=[package for package in find_packages() if package.startswith("mdgym")],
    package_data={
        "mdgym": [
            "envs/hawkins2DMap/pointclouds/*.txt",
        ]
    },
    python_requires=">=3.6",
    tests_require="",
    url="",
    version=VERSION,
    zip_safe=False,
)
