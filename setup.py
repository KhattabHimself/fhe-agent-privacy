"""Package setup for fhe-agent-privacy."""

from setuptools import find_packages, setup

setup(
    name="fhe-agent-privacy",
    version="0.1.0",
    description="Threat model and feasibility study of encrypted (PIR/FHE) tool-calling for AI agents",
    author="Mohamed",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "tenseal>=0.3.14",
        "pycryptodome>=3.19.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tabulate>=0.9.0",
        "memory-profiler>=0.61.0",
        "psutil>=5.9.0",
    ],
)
