"""
Setup script for CEM-RAG project
"""

from setuptools import setup, find_packages

setup(
    name="cem-rag",
    version="0.1.0",
    description="Compression Embedding Model for RAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "zstandard>=0.21.0",
    ],
    extras_require={
        "eval": [
            "mteb>=1.0.0",
        ],
    },
)

