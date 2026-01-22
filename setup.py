from setuptools import setup, find_packages, Extension

sm2_ext = Extension(
    "lunalib.core.sm2_c.sm2_ext",
    sources=[
        "lunalib/core/sm2_c/sm2_ext.c",
        "lunalib/core/sm2_c/sm2_bn.c",
        "lunalib/core/sm2_c/sm2_ec.c",
    ],
)

setup(
    name="lunalib",
    version="2.4.0",
    author_email="taellinglin@gmail.com",
    description="Cryptocurrency Ecosystem library (LunaLib) by Sanny and Ling Lin",
    long_description="A modular cryptocurrency ecosystem library including blockchain, wallet, mining, storage, and transaction management.",
    long_description_content_type="text/markdown",
    url="https://github.com/taellinglin/LunaLib",
    packages=find_packages(),
    ext_modules=[sm2_ext],
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
        extras_require={
            "gpu": ["cupy-cuda12x"],  # Optional: install with pip install lunalib[gpu]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)