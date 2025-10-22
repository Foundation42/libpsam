#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="libpsam",
    version="0.1.0",
    description="Fast, lightweight sequence prediction using learned token associations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Foundation42",
    author_email="",
    url="https://github.com/Foundation42/libpsam",
    packages=find_packages(),
    package_data={
        "psam": ["*.so", "*.dylib", "*.dll"],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp sequence-prediction language-model psam sparse-matrix machine-learning",
)
