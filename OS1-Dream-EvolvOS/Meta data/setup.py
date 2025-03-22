#!/usr/bin/env python3
"""
EvolvOS: Recursive Self-Evolving AI System
Setup Script
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="evolvos",
    version="0.1.0",
    description="Recursive Self-Evolving AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EvolvOS Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/evolvos",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "evolvos=main:main",
        ],
    },
) 