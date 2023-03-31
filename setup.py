"""Setups the project."""
import itertools
import re

from setuptools import find_packages, setup


print("Checking version...")
with open("gym/version.py") as file:
    full_version = file.read()
    matched_version = re.match(r'VERSION = "\d\.\d+\.\d+"', full_version).group()
    assert (matched_version == full_version), "Unexpected version: {}".format(full_version)
    VERSION = re.search(r"\d\.\d+\.\d+", full_version).group()

# Environment-specific dependencies.
extras = {
    "hawkins-2d-map": [""],
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
        # Python 3.6 is minimally supported (only with basic gym environments and API)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    name="multi-drone-gym",
    packages=[package for package in find_packages() if package.startswith("gym")],
    package_data={
        "gym": [
            "envs/hawkins-2d-map/pointclouds/*.txt",
        ]
    },
    python_requires=">=3.6",
    tests_require="",
    url="",
    version=VERSION,
    zip_safe=False,
)
