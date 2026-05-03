from pathlib import Path
from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="ndna",
    version="0.1.0",
    description="ndna: geometry-based analysis utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch==2.9.0",
        "torchvision==0.24.0",
        "torchaudio==2.9.0",
        "numpy==2.2.6",
        "transformers==4.57.3",
        "datasets==4.4.1",
        "accelerate==1.12.0",
        "huggingface-hub==0.36.0",
        "sentencepiece==0.2.1",
        "bitsandbytes==0.49.0; platform_system != 'Windows'",
        "tokenizers==0.22.0",
        "protobuf==6.33.2",
        "spacy==3.8.11",
        "similaritymeasures==1.4.0",
        "matplotlib==3.10.8",
        "plotly==6.5.0",
        "tqdm==4.67.1",
        "vllm==0.13.0",
        "peft==0.18.0",
        "trl==0.22.2",
    ],
    extras_require={
        "dev": [
            "pytest==9.0.2",
            "pytest-cov==7.0.0",
            "ipykernel==7.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)