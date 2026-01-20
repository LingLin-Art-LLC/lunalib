from setuptools import setup, find_packages

setup(
    name="lunalib",
    version="2.0.6",
    author_email="taellinglin@gmail.com",
    description="Cryptocurrency Ecosystem library (LunaLib) by Sanny and Ling Lin",
    long_description="A modular cryptocurrency ecosystem library including blockchain, wallet, mining, storage, and transaction management.",
    long_description_content_type="text/markdown",
    url="https://github.com/taellinglin/LunaLib",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        "numpy",
        "pytest",
        "pytest-benchmark",
        "pandas",
        "tqdm",
        "base58",
        "colorama",
        "msgpack",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)