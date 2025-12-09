from setuptools import setup, find_packages

setup(
    name="nanopore_signal_tokenizer",
    version="0.1.0",
    description="Tokenize Nanopore FAST5 signals into chunks with normalization and filtering",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ont-fast5-api",
        "numpy",
        "scipy",
        "pathos",
        "faiss-gpu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
