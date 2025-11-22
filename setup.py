from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Better requirements handling with fallback
requirements_path = "lunalib/requirements.txt"
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    # Fallback requirements if the file doesn't exist
    requirements = [
        "cryptography>=3.4",
        "requests>=2.25",
        # Add other core dependencies here
    ]

setup(
    name="lunalib",
    version="1.1.9",  # Consider using semantic versioning like "1.0.0"
    author="Ling Lin",
    author_email="taellinglin@gmail.com",
    description="A Complete Cryptocurrency Wallet and Mining System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taellinglin/LunaLib",
    project_urls={
        "Bug Tracker": "https://github.com/taellinglin/LunaLib/issues",
        "Documentation": "https://linglin.art/docs/luna-lib",
        "Source Code": "https://github.com/taellinglin/LunaLib",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",  # Added 3.11
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial",  # Added financial category
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "luna-wallet=lunalib.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lunalib": ["py.typed", "requirements.txt"],  # Include requirements.txt
    },
    # Optional: Add keywords for better discoverability
    keywords="cryptocurrency, blockchain, wallet, mining, bitcoin, ethereum, crypto",
    # Optional: Add license
    license="MIT",
)