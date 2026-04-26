from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="p-eagle",
    version="1.0.0",
    author="Juspay AI Team",
    author_email="ai@juspay.in",
    description="Production-grade P-EAGLE Parallel Speculative Decoding Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juspay/p-eagle",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "p-eagle-extract=p_eagle.scripts.extract_features:main",
            "p-eagle-train=p_eagle.scripts.train_drafter:main",
            "p-eagle-infer=p_eagle.scripts.run_inference:main",
            "p-eagle-prepare=p_eagle.data_preparation.data_manager:main",
        ],
    },
)
