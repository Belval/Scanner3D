from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scanner3d",
    version="0.1.0",
    description="A Python module for 3D scanning using SR300/D435i cameras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Belval/scanner3d",
    author="Edouard Belval",
    author_email="edouard@belval.org",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="3d scanner",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "pyrealsense",
        "open3d",
        "opencv-python",
        "numpy",
        "scikit-learn",
        "hdbscan",
        "seaborn",
    ],
)