from setuptools import setup, find_packages, Extension
from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).resolve().parent

sm2_ext = Extension(
    "lunalib.core.sm2_c.sm2_ext",
    sources=[
        "lunalib/core/sm2_c/sm2_ext.c",
        "lunalib/core/sm2_c/sm2_bn.c",
        "lunalib/core/sm2_c/sm2_ec.c",
    ],
    include_dirs=[str(BASE_DIR / "lunalib" / "core" / "sm2_c")],
)

def _sm3_compile_args():
    args = []
    if os.getenv("LUNALIB_AVX512", "0") == "1":
        if os.name == "nt":
            args.append("/arch:AVX512")
        else:
            args.extend(["-mavx512f", "-O3"])
    if os.getenv("LUNALIB_AVX2", "0") == "1":
        if os.name == "nt":
            args.append("/arch:AVX2")
        else:
            args.extend(["-mavx2", "-O3"])
    return args


def _sm3_define_macros():
    if os.getenv("LUNALIB_AVX512", "0") == "1":
        return [("LUNALIB_AVX512", "1")]
    if os.getenv("LUNALIB_AVX2", "0") == "1":
        return [("LUNALIB_AVX2", "1")]
    return []


sm3_ext = Extension(
    "lunalib.core.sm3_c.sm3_ext",
    sources=[
        "lunalib/core/sm3_c/sm3_ext.c",
    ],
    extra_compile_args=_sm3_compile_args(),
    define_macros=_sm3_define_macros(),
)

setup(
    name="lunalib",
    version="2.6.1",
    author_email="taellinglin@gmail.com",
    description="Cryptocurrency Ecosystem library (LunaLib) by Sanny and Ling Lin",
    long_description="A modular cryptocurrency ecosystem library including blockchain, wallet, mining, storage, and transaction management.",
    long_description_content_type="text/markdown",
    url="https://github.com/taellinglin/LunaLib",
    packages=find_packages(),
    ext_modules=[sm2_ext, sm3_ext],
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
            "gpu": ["cupy-cuda12x; python_version < '3.12'"],  # Optional: pip install lunalib[gpu]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)